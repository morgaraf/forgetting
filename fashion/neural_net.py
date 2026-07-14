# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Dropout,
    Dense,
    LeakyReLU,
    Flatten,
    Reshape,
    UpSampling2D,
    BatchNormalization,
    # LayerNormalization,
    SpatialDropout2D,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Mean, CategoricalAccuracy, RootMeanSquaredError
import constants
import dataset

epochs = 300
patience = 10
truly_training_percentage = 0.80


def conv_block(entry, layers, filters, dropout, first_block=False, pooling=True):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(
                kernel_size=3,
                padding='same',
                activation='relu',
                filters=filters,
            )(entry)
            first_block = False
        else:
            conv = Conv2D(
                kernel_size=3, padding='same', activation='relu', filters=filters
            )(entry)
        entry = BatchNormalization()(conv)
    output = (
        entry
        if not pooling
        else MaxPool2D(pool_size=2, strides=2, padding='same')(entry)
    )
    output = SpatialDropout2D(dropout)(output)
    return output


# The number of layers defined in get_encoder.
encoder_nlayers = 40


def get_encoder(domain):
    dropout = 0.1
    input_data = Input(shape=(dataset.rows, dataset.columns, 1))
    filters = domain // 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 2, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)

    # --- THE FEATURE BOOSTER ---
    # We add a 2*domain-filter block here to capture fine-grained textures.
    # But we DO NOT increase the final domain size.
    # dropout *= 2.0
    # output = Conv2D(2 * domain, kernel_size=3, padding='same', activation='relu')(
    #     output
    # )
    # output = BatchNormalization()(output)
    # output = SpatialDropout2D(0.2)(output)
    # --------------------------------------

    output = Flatten()(output)
    # output = Dense(constants.domain, name='domain_layer')(output)
    # output = LayerNormalization()(output)
    return input_data, output


def get_decoder(domain):
    n = int(math.log2(domain))
    remainer = 3 if (n % 2 != 0) else 2
    initial_divisor = 2 * remainer
    iter_divisor = 2 ** ((n - remainer) // 2 - 1)

    input_mem = Input(shape=(domain,))
    # Which is going to be multiplied by two by each Conv2DTranspose layer in the loop.
    width = dataset.columns // 4
    filters = domain // initial_divisor
    dense = Dense(width * width * filters, activation='relu')(input_mem)
    output = Reshape((width, width, filters))(dense)
    dropout = 0.1
    for i in range(2):
        filters = filters // iter_divisor
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(filters, (3, 3), padding='same')(output)
        output = BatchNormalization()(output)  # Optional in decoder
        output = LeakyReLU(alpha=0.2)(output)
        output = SpatialDropout2D(dropout)(output)
        dropout /= 2.0
    output = Conv2D(
        filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same'
    )(output)
    return input_mem, output


# The number of layers defined in get_classifier.
classifier_nlayers = 6


def get_classifier(domain):
    input_mem = Input(shape=(domain,))
    # Uses LeakyReLU or ELU, as they allow negative values to pass through,
    # so the classifier can "see" the full latent space.
    dense = Dense(4 * domain)(input_mem)
    dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    dense = Dense(2 * domain)(drop)
    dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    # dense = Dense(domain)(drop)
    # dense = LeakyReLU(negative_slope=0.1)(dense)
    # drop = Dropout(0.2)(dense)
    # dense = Dense(domain // 2)(drop)
    # dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    classification = Dense(
        constants.n_labels, activation='softmax', name='classified'
    )(drop)
    return input_mem, classification


def train_network(prefix):
    confusion_matrix = np.zeros((constants.network_labels, constants.network_labels))
    histories = []
    strategy = tf.distribute.MirroredStrategy()
    for fold in range(constants.n_folds):
        print(f'FOLD: {fold}')
        print('Getting the dataset ready...')
        training_gen = dataset.get_training(fold, categorical=True)
        # No shuffling is needed for validation nor testing.
        validating_gen = dataset.get_validating(fold, categorical=True)
        testing_gen = dataset.get_testing(fold, categorical=True)
        predict_gen = dataset.get_testing(fold, predict_only=True)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        with strategy.scope():
            domain = constants.domain
            print('Building and compiling the neural network...')
            input_data = Input(shape=(dataset.rows, dataset.columns, 1))
            input_enc, output_enc = get_encoder(domain)
            input_class, output_class = get_classifier(domain)
            input_dec, output_dec = get_decoder(domain)

            encoder = Model(input_enc, output_enc, name='encoder')
            encoder.summary()
            classifier = Model(input_class, output_class, name='classifier')
            classifier.summary()
            decoder = Model(input_dec, output_dec, name='decoder')
            decoder.summary()
            encoded = encoder(input_data)
            # decoded = decoder(encoded)
            classified = classifier(encoded)

            decoder_weight_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            warmup_cb = DecoderWeightScheduler(decoder_weight_var, linear_warmup)

            # model = Model(
            #     inputs=input_data,
            #     outputs={'classifier': classified, 'decoder': decoded},
            # )
            # 2. Instantiate the PerceptionModel
            model = PerceptionModel(
                encoder=encoder,
                classifier=classifier,
                decoder=decoder,
                num_classes=constants.network_labels,
                latent_dim=constants.domain,
                center_loss_weight=0.1,  # You can adjust this weight as needed
            )
            model.decoder_weight_var = decoder_weight_var
            model.compile(
                loss={
                    'classifier': 'categorical_crossentropy',
                    'decoder': 'mean_squared_error',
                },
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=1e-3
                ),  # Learning rate for a batch size of 2048
                loss_weights={'classifier': 1, 'decoder': decoder_weight_var},
                metrics={'classifier': 'accuracy', 'decoder': rmse},
            )
            model.summary()

            full_classifier = Model(
                inputs=input_data, outputs=classified, name='full_classifier'
            )
            # autoencoder = Model(inputs=input_data, outputs=decoded, name='autoencoder')

        print('Training the neural network...')
        early_stopping = EarlyStopping(
            monitor='val_classifier_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=2,
        )

        lr_reducer = ReduceLROnPlateau(
            monitor='val_classifier_accuracy',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            mode='max',
            verbose=2,
        )

        training_history_object = model.fit(
            training_gen,
            # batch_size=constants.batch_size,
            epochs=epochs,
            validation_data=validating_gen,
            callbacks=[early_stopping, lr_reducer, warmup_cb],
            verbose=2,
        )
        # Extracts only the history of the training from the Keras History object.
        # Extract the actual dictionary and convert NumPy values to Python floats
        training_history = {
            k: [float(val) for val in v]
            for k, v in training_history_object.history.items()
        }
        # The history returned by model.evaluate is a dictionary of metric names to values,
        # simpler than the one returned by model.fit.
        evaluation_history_dict = model.evaluate(testing_gen, return_dict=True)
        evaluation_history = {k: float(v) for k, v in evaluation_history_dict.items()}
        predicted_labels = np.argmax(full_classifier.predict(predict_gen), axis=1)
        # Retrieve True Labels directly from HDF5 using generator indices
        true_labels = predict_gen.get_all_labels()
        print('Creating the confusion matrix...')
        confusion_matrix += tf.math.confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes=constants.network_labels,
        )
        print('Saving everything needed for the future...')
        encoder.save(constants.encoder_filename(prefix, fold))
        decoder.save(constants.decoder_filename(prefix, fold))
        classifier.save(constants.classifier_filename(prefix, fold))
        # Saving the centers for later inspection
        np.save(constants.centers_filename(prefix, fold), model.centers.numpy())
        name = constants.classification_name()
        prediction_filename = constants.data_filename(name, es=None, fold=fold)
        np.save(prediction_filename, predicted_labels)
        history = {
            'fold': fold,
            'training': training_history,
            'evaluation': evaluation_history,  # Already a dict due to return_dict=True
        }
        histories.append(history)
    history_record = {
        'metadata': {
            'batch_size': constants.batch_size,
            'epochs': epochs,
            'n_folds': constants.n_folds,
        },
        'results': histories,
    }
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    return history_record, confusion_matrix / totals


def obtain_features(model_prefix, features_prefix, labels_prefix):
    for fold in range(constants.n_folds):
        # Load the encoder
        filename = constants.encoder_filename(model_prefix, fold)
        model = tf.keras.models.load_model(filename)

        # 1. Get Generators (which replace the raw data arrays)
        # We set predict_only=True so the generator returns ONLY images for model.predict.
        fill_gen = dataset.get_filling(fold, predict_only=True)
        test_gen = dataset.get_testing(fold, predict_only=True)
        settings = [
            (fill_gen, constants.filling_suffix),
            (test_gen, constants.testing_suffix),
        ]

        for gen, suffix in settings:
            print(
                f'Generating features for {suffix} (batch-by-batch to prevent OOM)...'
            )

            features = []
            # Reset generator if it's a reusable iterator/sequence
            for batch in gen:
                # Safe unwrap in case the generator returns a tuple (x, y) or just x
                x_batch = batch[0] if isinstance(batch, (tuple, list)) else batch

                # Predict on a single batch and immediately force it into CPU memory as a NumPy array
                batch_feats = model.predict_on_batch(x_batch)
                features.append(np.array(batch_feats))

            # Safely concatenate all batches together entirely on the CPU
            features = np.concatenate(features, axis=0)
            labels = gen.get_all_labels()
            features_filename = constants.shared_data_filename(
                features_prefix + suffix, fold
            )
            labels_filename = constants.shared_data_filename(
                labels_prefix + suffix, fold
            )

            print('Saving features and labels ...')
            np.save(features_filename, features)
            np.save(labels_filename, labels)


class PerceptionModel(Model):
    def __init__(
        self,
        encoder,
        classifier,
        decoder,
        num_classes,
        latent_dim,
        center_loss_weight=0.1,
        alpha=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.center_loss_weight = center_loss_weight
        self.alpha = alpha

        # Centers bank
        self.centers = self.add_weight(
            name='centers',
            shape=(num_classes, latent_dim),
            initializer='zeros',
            trainable=False,
        )

        # 1. Explicit Trackers (Bypassing Keras dictionary mapping entirely)
        self.loss_tracker = Mean(name='loss')
        self.class_loss_tracker = Mean(name='classifier_loss')
        self.recon_loss_tracker = Mean(name='decoder_loss')
        self.center_loss_tracker = Mean(name='center_loss')

        # Explicit Metrics matching your previous naming
        self.acc_tracker = CategoricalAccuracy(name='classifier_accuracy')
        self.rmse_tracker = RootMeanSquaredError(name='decoder_root_mean_squared_error')

        # Loss objects
        self.class_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.recon_loss_fn = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
        # Tell Keras to pull from our explicit trackers for the progress bar
        return [
            self.loss_tracker,
            self.class_loss_tracker,
            self.recon_loss_tracker,
            self.center_loss_tracker,
            self.acc_tracker,
            self.rmse_tracker,
        ]

    def call(self, inputs):
        latent = self.encoder(inputs)
        return {'classifier': self.classifier(latent), 'decoder': self.decoder(latent)}

    def train_step(self, data):
        x, y_wrapped = data

        # Unwrap the labels from Keras's auto-dictionary
        if isinstance(y_wrapped, dict):
            y_labels = y_wrapped.get('classifier', y_wrapped)
        elif isinstance(y_wrapped, (list, tuple)):
            y_labels = y_wrapped[0]
        else:
            y_labels = y_wrapped

        # Get the current dynamic decoder weight
        d_weight = getattr(self, 'decoder_weight_var', 1.0)

        with tf.GradientTape() as tape:
            # 2. Forward Pass
            latent_features = self.encoder(x, training=True)
            pred_class = self.classifier(latent_features, training=True)
            pred_recon = self.decoder(latent_features, training=True)

            # Uses the instantiated loss objects
            class_loss = self.class_loss_fn(y_labels, pred_class)
            recon_loss = self.recon_loss_fn(x, pred_recon)

            # Center Loss
            label_indices = tf.argmax(y_labels, axis=1)
            batch_centers = tf.gather(self.centers, label_indices)
            center_loss = tf.reduce_mean(tf.square(latent_features - batch_centers))

            # Total loss using your dynamic weights
            total_loss = (
                class_loss
                + (d_weight * recon_loss)
                + (self.center_loss_weight * center_loss)
            )

        # 4. Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 5. Safe Center Update
        delta = batch_centers - latent_features
        diff_sum = tf.math.unsorted_segment_sum(delta, label_indices, self.num_classes)
        counts = tf.math.bincount(
            label_indices, minlength=self.num_classes, dtype=tf.float32
        )
        counts = tf.reshape(counts, [-1, 1])
        diff_mean = diff_sum / (counts + 1.0)
        self.centers.assign_sub(self.alpha * diff_mean)

        # 6. Update Explicit Metrics
        self.loss_tracker.update_state(total_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.center_loss_tracker.update_state(center_loss)
        self.acc_tracker.update_state(y_labels, pred_class)
        self.rmse_tracker.update_state(x, pred_recon)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_wrapped = data

        # Unwrap the labels from Keras's auto-dictionary
        if isinstance(y_wrapped, dict):
            y_labels = y_wrapped.get('classifier', y_wrapped)
        elif isinstance(y_wrapped, (list, tuple)):
            y_labels = y_wrapped[0]
        else:
            y_labels = y_wrapped

        # Validation Pass
        latent_features = self.encoder(x, training=False)
        pred_class = self.classifier(latent_features, training=False)
        pred_recon = self.decoder(latent_features, training=False)

        class_loss = self.class_loss_fn(y_labels, pred_class)
        recon_loss = self.recon_loss_fn(x, pred_recon)

        # Update Validation Metrics
        self.class_loss_tracker.update_state(class_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.acc_tracker.update_state(y_labels, pred_class)
        self.rmse_tracker.update_state(x, pred_recon)

        # Return everything except total loss and center loss for validation
        return {
            m.name: m.result()
            for m in self.metrics
            if m.name not in ['loss', 'center_loss']
        }


class DecoderWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, weight_var, schedule_fn):
        super(DecoderWeightScheduler, self).__init__()
        self.weight_var = weight_var
        self.schedule_fn = schedule_fn

    def on_epoch_begin(self, epoch, logs=None):
        new_weight = self.schedule_fn(epoch)
        # Use assign to update the tensor value without re-compiling
        self.weight_var.assign(new_weight)
        print(f'\n - current_decoder_weight: {self.weight_var.numpy():.4f}')


def linear_warmup(epoch):
    start_weight = 0.0
    end_weight = 10.0
    warmup_epochs = 160  # Reach full weight by epoch, then keep it constant

    if epoch < warmup_epochs:
        return start_weight + (end_weight - start_weight) * (epoch / warmup_epochs)
    return end_weight
