#!/bin/bash
eval "$(conda shell.bash hook)"
# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages
conda create -n MuseTalk python=3.10
conda activate MuseTalk

pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

rm -rf /venv/MuseTalk/lib/python3.10/site-packages/mmcv*

pip uninstall -y mmcv mmcv-full mmcv-lite
pip install --upgrade pip setuptools wheel
pip install "setuptools==60.2.0"
pip install chumpy==0.70 --no-build-isolation
pip install mmpose openmim --no-build-isolation
pip install --no-cache-dir -U openmim
mim install mmengine --no-build-isolation
pip install mmcv==2.0.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
  --no-cache-dir
mim install "mmdet==3.1.0" --no-build-isolation
mim install "mmpose==1.1.0" --no-build-isolation
#pip install -U "huggingface_hub[cli]"
#pip install gdown

sudo apt-get install libx264-dev
conda uninstall ffmpeg
conda install -c conda-forge ffmpeg
conda update ffmpeg

export FFMPEG_PATH=$(which ffmpeg)

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
huggingface-cli download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
huggingface-cli download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth
  
  
sh inference.sh v1.5 normal

echo "✅ Installation successful!"
