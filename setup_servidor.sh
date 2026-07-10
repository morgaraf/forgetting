#!/bin/bash
# ==============================================================================
# PROYECTO DELFÍN - Script de configuración para servidor de la universidad
# Ejecutar: bash setup_servidor.sh
# ==============================================================================

echo "============================================"
echo "  CONFIGURACIÓN DEL SERVIDOR - PROYECTO DELFÍN"
echo "============================================"

# 1. Descargar e instalar Anaconda (Linux)
if ! command -v conda &> /dev/null; then
    echo ""
    echo "1. Descargando Anaconda..."
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh
    bash anaconda.sh -b -p $HOME/anaconda3
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "   -> Anaconda instalada. Cierra y abre la terminal, luego ejecuta este script de nuevo."
    exit 0
else
    echo "1. Anaconda ya está instalada."
fi

# 2. Crear ambiente conda
echo ""
echo "2. Creando ambiente 'delfin'..."
conda create -n delfin python=3.10 -y
conda activate delfin

# 3. Instalar dependencias
echo ""
echo "3. Instalando dependencias..."
pip install -r requirements.txt

# 4. Verificar GPU
echo ""
echo "4. Verificando GPU..."
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPUs encontradas:', len(gpus)); [print('  -', g) for g in gpus]"

echo ""
echo "============================================"
echo "  ¡CONFIGURACIÓN COMPLETA!"
echo "  Para ejecutar el experimento:"
echo "  conda activate delfin"
echo "  python experimento_semana5.py"
echo "============================================"
