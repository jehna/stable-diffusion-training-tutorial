This is a tutorial to get teach you how to teach a custom face to Stable
Diffusion using Google Cloud Platform (GCP).

## Prerequisites

* GCP account with a spot A100 GPU instance limit >= 1

## Setup the VM

(These steps you'll only need to do once)

Create a Compute Engine (GCP) instance:

* Choose GPU -> NVIDIA A100 40GB machine family
* Choose an OS with pre-installed PyTorch (via "Switch Image" button),
  * e.g.  `Debian 10 based Deep Learning VM for PyTorch CPU/GPU with CUDA 11.3 M97`
* Choose a storage size of at least 100gb
* Choose Advanced options -> Management -> Availability policies -> Spot (if you want to save money)

1. SSH into the instance

The instance asks "Would you like to install the Nvidia driver?" on the first login. Choose "y".

2. Clone Dreambooth repo

```bash
git clone https://github.com/XavierXiao/Dreambooth-Stable-Diffusion.git
cd Dreambooth-Stable-Diffusion/
```

3. Create the conda environment

```bash
conda env create -f environment.yaml
```

4. Download the Stable Diffusion model

Go to https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main
and download the model (sd-v1-4.ckpt). Copy the model over to the VM.

```bash
scp sd-v1-4.ckpt ec2-user@<VM_IP>:~/Dreambooth-Stable-Diffusion/
```

Note that you need to be logged in to HuggingFace to download the model (create
a free account if you don't have one).

5. Create an image for training images and regularization images

```bash
mkdir training-images
mkdir regularization-images
```

6. You're done! Log out and now you can start training the model with your own images

## Train the model

1. Crop your training images (5-8 pictures) to squares with 512x512 pixels dimensions, with the face in the center

2. Move your training images into the VM

```bash
scp training-images/* $HOST:~/Dreambooth-Stable-Diffusion/training-images/
```

Make sure you don't show teeth in the training images — Stable Diffusion is not
that good in drawing teeth and having teeth in the training images will make
Stable Diffusion to draw teeth in the final image.

3. SSH into the instance

4. Ectivate Conda environment

```bash
cd Dreambooth-Stable-Diffusion/
conda activate ldm
```

5. Generate 10 regularization images

```bash
python scripts/stable_txt2img.py \
  --prompt "photo of <subject>" \
  --n_samples 10 \
  --skip_grid \
  --seed $RANDOM \
  --outdir regularization-images \
  --ckpt sd-v1-4.ckpt
```

6. Train the model

```bash
python main.py \
  --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
  -t \
  --actual_resume sd-v1-4.ckpt \
  -n a_m \
  --gpus 0, \
  --seed $RANDOM \
  --data_root training-images \
  --reg_data_root regularization-images/samples \
  --class_word '<subject>'
```

7. Run inference

```bash
python scripts/stable_txt2img.py \
  --n_samples 1 \
  --n_iter 1 \
  --ckpt logs/training-images<TIMESTAMP>/checkpoints/last.ckpt \
  --skip_grid \
  --seed $RANDOM \
  --prompt 'sks <subject>'
```

8. Load images back to your own machine

```bash
scp $HOST:~/Dreambooth-Stable-Diffusion/outputs/txt2img-samples/samples/* ./
```

## Download the .ckpt to local use

```bash
scp $HOST:~/Dreambooth-Stable-Diffusion/logs/training-images<TIMESTAMP>/checkpoints/last.ckpt ./
```