import gradio as gr
#import gradio.helpers
import torch
import os
from glob import glob
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

import uuid
import random
from huggingface_hub import hf_hub_download
import tomesd

#gradio.helpers.CACHED_FOLDER = '/data/cache'

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

tomesd.apply_patch(pipe, ratio=0.5)  
# compiling causes sampling to crash for me atm
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

max_64_bit_int = 2**63 - 1

def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = False,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt",
    cond_aug: float = 0.02,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = "outputs",
):
    print("called sample")
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    if(randomize_seed):
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    frames = pipe(image, 
                  decode_chunk_size=decoding_t, 
                  generator=generator, 
                  motion_bucket_id=motion_bucket_id, 
                  noise_aug_strength=0.1,
                  num_frames=14,
                  ).frames[0]
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)
    
    return video_path, seed

def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image



def run(image_path):
    image = resize_image(Image.open(image_path))
    video_path, seed = sample(image)
    return video_path, seed

if __name__ == "__main__":

    run("/mnt/newdrive/svd-playground/assets/output.png")