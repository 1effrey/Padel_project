#!/usr/bin/env bash
# scripts/runpod_setup.sh
# One-shot setup for training the padel BALL detector (TrackNet heatmap) on a RunPod GPU.
# Use a RunPod "PyTorch" template (torch + CUDA already installed). Run this on the POD.
set -euo pipefail

# --- 1. Code + configs + label CSVs (all tracked in git) -----------------------------
#   The two VIDEOS are NOT in git (too big) -- pull them with gdown (see step 3).
#   This clones 'main', which already has train_ball.py + ball_detector.py + the labels.
#   (If/when you push your 2D branch, run afterwards:  git checkout <your-2d-branch> .)
git clone https://github.com/1effrey/Padel_project.git
cd Padel_project

# --- 2. Python deps ------------------------------------------------------------------
#   torch + CUDA come with the RunPod PyTorch image -- verify, do NOT reinstall torch.
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
#   Headless box -> use opencv-python-headless (no GUI libs). The rest are light.
pip install --no-input ultralytics supervision opencv-python-headless numpy matplotlib polars tqdm
#   If 'import cv2' ever complains about libGL on the pod:
#       apt-get update && apt-get install -y libgl1 libglib2.0-0

# --- 3. Videos via gdown from Google Drive -- the FULL videos, NOT the clips! ---------
#   IMPORTANT: the labels (output/ball_labels_side-*-full-vid.csv) index the FULL videos;
#   the short CLIPS will NOT line up with them. Upload side-1-full-vid.mp4 (~324 MB) and
#   side-2-full-vid.mp4 (~1002 MB) to Drive, then pull THOSE here.
#   For each: in Drive -> Share -> "Anyone with the link"; the URL is
#   drive.google.com/file/d/<FILE_ID>/view  -- paste that <FILE_ID> into the lines below.
pip install --no-input gdown
gdown "https://drive.google.com/uc?id=<FILE_ID_side1_FULL>" -O side-1-full-vid.mp4
gdown "https://drive.google.com/uc?id=<FILE_ID_side2_FULL>" -O side-2-full-vid.mp4
ls -lh side-1-full-vid.mp4 side-2-full-vid.mp4   # expect ~324 MB and ~1002 MB

# --- 4. Train (gated: verify the loop, then the real run, then measure) --------------
python train_ball.py --smoke                                              # random data: loop OK?
python train_ball.py --dry-run --config config-side1.json --config2 config-side2.json  # your data, no real train
python train_ball.py --config config-side1.json --config2 config-side2.json            # the real training run
python main.py --precision --config config-side1.json                     # the quality GATE (held-out split)
echo "Done. Read output/ for the trained weights + precision metrics; copy weights back to your laptop."
