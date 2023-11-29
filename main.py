import torch
from omegaconf import OmegaConf
from requests import get 
from safetensors import safe_open
from PIL import Image
from io import BytesIO
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_video import (
    StableDiffusionVideoPipeline
)
from diffusers.models import (
    UNetSpatioTemporalConditionModel,
    AutoencoderKLTemporalDecoder
)
from diffusers.schedulers import (
    EulerDiscreteScheduler
)
from convert_svd_to_diffusers import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint
)

# CONFIGURABLES
IMAGE_SIZE = 768 # For the UNet and VAE config, not the image we're using to generate
CONFIG_URL = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/svd.yaml"
CHECKPOINT_PATH = "/mnt/newdrive/svd-playground/checkpoints/svd.safetensors"
# CHECKPOINT_PATH = "/mnt/newdrive/svd-playground/checkpoints/svd_xt.safetensors"
CACHE_DIR = "/home/benjamin/.cache/enfugue/cache"
TEST_IMAGE = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/assets/test_image.png"

# CKPT
original_config = OmegaConf.create(get(CONFIG_URL).text)
state_dict = {}
with safe_open(CHECKPOINT_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

# VAE
vae_params = original_config.model.params.first_stage_config.params.decoder_config.params
block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
up_block_types = ["UpBlockTemporalDecoder"] * len(block_out_channels)

vae_config = { 
    "sample_size": IMAGE_SIZE,
    "in_channels": vae_params.in_channels,
    "out_channels": vae_params.out_ch,
    "down_block_types": tuple(down_block_types),
    # "up_block_types": tuple(up_block_types),
    "block_out_channels": tuple(block_out_channels),
    "latent_channels": vae_params.z_channels,
    "layers_per_block": vae_params.num_res_blocks,
}   

vae = AutoencoderKLTemporalDecoder(**vae_config)
vae_state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config)
result = vae.load_state_dict(vae_state_dict, strict=False) # Has to be non-strict, makes me thing my VAE is wrong
print("result missing keys", result.missing_keys)
print("result unexpected keys", result.unexpected_keys)

# UNET
config = create_unet_diffusers_config(original_config, image_size=IMAGE_SIZE)
unet_state_dict = convert_ldm_unet_checkpoint(state_dict, config, path=CHECKPOINT_PATH)
unet = UNetSpatioTemporalConditionModel.from_config(config)
unet.load_state_dict(unet_state_dict)

# SCHEDULER
scheduler = EulerDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    use_karras_sigmas=True,
)

# VISION
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=torch.float16,
#     cache_dir=CACHE_DIR
)
feature_extractor = CLIPImageProcessor()

# PIPE
pipe = StableDiffusionVideoPipeline(
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
)
pipe = pipe.to("cuda", dtype=torch.float16)

# INITIAL IMAGE
image = Image.open(BytesIO(get(TEST_IMAGE).content)).convert("RGB")
width, height = image.size
width = (width // 8) * 8
height = (height // 8) * 8
image = image.resize((width, height))

# INVOKE
out = pipe(
    image=image,
    width=width,
    height=height,
    num_frames=14,
    num_inference_steps=12, # I've tried up to 150 steps
    decoding_t=1,
    output_type="pil"
)

# SAVE
frames = out["frames"][0]
frames[0].save("./output.gif", loop=0, duration=125, save_all=True, append_images=frames[1:])