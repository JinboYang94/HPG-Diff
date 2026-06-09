echo "=========================================="
echo "QLoRA diffusion fine-tuning"
echo "=========================================="

echo "Checking Python environment..."
python --version

echo "Checking CUDA..."
nvidia-smi

echo "Installing dependencies..."
pip install -r ../requirements.txt

echo "Starting training..."
python LoRA_diffusion.py

echo "Training complete."
