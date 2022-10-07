This is a tutorial to get teach you how to teach a custom face to Stable
Diffusion using AWS P3 instance.

Note that AWS has long wait times for P3 instances, and then instance types are
not that cost effective for large models as with GCP. Because of this, we need
to use a different repo for training on AWS that is compatible with running the
model with 16gb of memory per GPU.

## Prerequisites

* AWS account with a GPU instance limit >= 1
* HuggingFace account's API key (create a free account if you don't have one)

## Setup the VM

(These steps you'll only need to do once)

Create a EC2 (AWS) instance.

* Choose an OS with pre-installed PyTorch like "Deep Learning AMI GPU PyTorch 1.12.1"
* Choose an instance with at least 1 GPU with 16gb memory (like p3)
* Choose a storage size of at least 100gb

1. SSH into the instance

2. Clone ShivamShrirao/diffusers repo

```bash
git clone https://github.com/ShivamShrirao/diffusers.git
cd diffusers/examples/dreambooth
```

3. Install required packages

```bash
pip3 install git+https://github.com/ShivamShrirao/diffusers.git
pip3 install -U -r requirements.txt
pip3 install bitsandbytes
accelerate config # No distributed training, no deepspeed, no fp16/bf16
```

4. Create an image for training images and output images

```bash
mkdir training-images
mkdir output-model
mkdir class-images
mkdir output-images
```

5. Create a script for inference and converting to .ckpt

```bash
cat << EOF > inference.py
from diffusers import StableDiffusionPipeline
import torch
import time
import sys
import re

model_id = "output-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# load prompt from arguments
prompt = sys.argv[1]
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("output-images/{}_{}.png".format(re.sub(r'\W+', '_', prompt).lower(), int(time.time())))
EOF
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
```

5. You're done! Log out and now you can start training the model with your own images

## Train the model

1. Crop your training images (5-8 pictures) to squares with 512x512 pixels dimensions, with the face in the center

2. Move your training images into the VM

```bash
scp training-images/* ec2-user@<VM_IP>:~/diffusers/examples/dreambooth/training-images/
```

Make sure you don't show teeth in the training images — Stable Diffusion is not
that good in drawing teeth and having teeth in the training images will make
Stable Diffusion to draw teeth in the final image.

3. SSH into the instance

4. Train the model

```bash
cd diffusers/examples/dreambooth
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4' \
  --instance_data_dir=training-images \
  --class_data_dir=class-images \
  --output_dir=output-model \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="sks <subject>" \
  --class_prompt="a photo of <subject>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

5. Run inference

```bash
python3 inference.py "a photo of sks <subject>"
```

6. Load images back to your own machine

```bash
scp ec2-user@<VM_IP>:~/diffusers/examples/dreambooth/output-images/* ./
```

## Convert to .ckpt

1. Run the conversion script

```bash
python3 convert_diffusers_to_original_stable_diffusion.py --model_path output-model  --checkpoint_path output-model/latest.ckpt
```

2. Download the model from the VM

```bash
scp ec2-user@<VM_IP>:~/diffusers/examples/dreambooth/output-model/latest.ckpt ./
```