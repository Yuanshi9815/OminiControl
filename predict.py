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


MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
FLUX_DEV_PATH = Path("FLUX.1-dev")


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
        ).to("cuda")


    def predict(
        self,
        prompt: str = Input(
            description="Input prompt.",
            default="On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat.",
        ),
        image: Path = Input(description="Input image"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
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
        height: int = Input(
            description="Height of the output image", default=512
        ),
        width: int = Input(
            description="Width of the output image", default=512
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        self.pipe.load_lora_weights(lora, weight_name=weight_name, adapter_name="cartoon")
        self.pipe.set_adapters("cartoon")
        self.pipe.to("cuda")

        image = Image.open(str(image)).convert("RGB").resize((width, height))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)
        generator = torch.Generator("cuda").manual_seed(seed)
        condition = Condition(model.split("_")[0], image)

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

        self.pipe.delete_adapters('cartoon')
        out_path = "/tmp/out.png"
        result_img.save(out_path)
        return Path(out_path)
        