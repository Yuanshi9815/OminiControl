# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything
from py_real_esrgan.model import RealESRGAN


MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
FLUX_DEV_PATH = Path("FLUX.1-dev")
ESRGAN_WEIGHTS_PATH = Path("weights/RealESRGAN_x4.pth")

upscale_factor = 2

device = 'cuda'

def download_base_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        print("Loading Flux dev pipeline")
        if not FLUX_DEV_PATH.exists():
            download_base_weights(MODEL_URL_DEV, Path("."))

        self.pipe = FluxPipeline.from_pretrained(
            "FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.upscaler = RealESRGAN(device, scale=upscale_factor)
        self.upscaler.load_weights(ESRGAN_WEIGHTS_PATH, download=True)


    def predict(
        self,
        prompt: str = Input(
            description="Input prompt.",
            default="A girl cartoon character in a white background. She is looking right, and running.",
        ),
        image: Path = Input(description="Input image"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        lora: str = Input(
            description="LoRA model to use.",
            default="saquiboye/oye-cartoon"
        ),
        weight_name: str = Input(
            description="LoRA weight to use.",
            default="pytorch_lora_weights.safetensors"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        position_delta: int = Input(
            description="Position delta for the condition", default=-16
        ),
        width: int = Input(
            description="Width of the output image", default=None
        ),
        height: int = Input(
            description="Height of the output image", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        print(f"Generating image with prompt: {prompt}")

        can_crop = False
        if height is None or width is None:
          height = 512
          width = 512
          can_crop = True
        else:
          height = int((height//16) * 16)
          width = int((width//16) * 16)

        self.pipe.load_lora_weights(lora, weight_name=weight_name, adapter_name="cartoon")

        image = Image.open(str(image)).convert("RGB")

        # Crop the image to a square
        _width, _height = image.size
        if _width != _height and can_crop:
            size = min(_width, _height)
            left = (_width - size) // 2
            top = (_height - size) // 2
            right = (_width + size) // 2
            bottom = (_height + size) // 2
            image = image.crop((left, top, right, bottom))
        
        image = image.resize((width, height))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)
        generator = torch.Generator("cuda").manual_seed(seed)
        condition = Condition('cartoon', condition=image, position_delta=[0, position_delta])

        result_img = generate(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        result_img = self.upscaler.predict(result_img)

        self.pipe.delete_adapters('cartoon')
        out_path = "/tmp/out.png"
        result_img.save(out_path)
        return Path(out_path)
        