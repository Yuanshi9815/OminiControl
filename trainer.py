
import os
import shutil
from zipfile import ZipFile, is_zipfile
from cog import BaseModel, Input, Path, Secret  # pyright: ignore
import yaml
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import lightning as L

from src.train.data import (
    ImageConditionDataset,
    Subject200KDateset,
    CartoonDataset
)
from src.train.model import OminiModel
from src.train.callbacks import TrainingCallback


INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")

class TrainingOutput(BaseModel):
    weights: Path

def get_config(config_path):
    assert config_path is not None, "Config path is required"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train(
    dataset_repo: str = Input(
        description="Hugging Face dataset id. For example, saquiboye/oye-cartoon.",
        default='saquiboye/oye-cartoon',
    ),
    config_path: str = Input(
        description="Config path. It must exist in the repository. Example: train/config/cartoon_1024.yaml",
        default='train/config/cartoon_1024.yaml',
    ),
    steps: int = Input(
        description="Number of training steps. Recommended range 500-4000",
        ge=3,
        le=6000,
        default=1000,
    ),
    learning_rate: float = Input(
        description="Learning rate, if you’re new to training you probably don’t need to change this.",
        default=4e-4,
    ),
    resolution: str = Input(
        description="Image resolutions for training", default="512,768,1024"
    ),
    lora_rank: int = Input(
        description="Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=4,
        ge=1,
        le=128,
    ),
    caption_dropout_rate: float = Input(
        description="Advanced setting. Determines how often a caption is ignored. 0.05 means for 5% of all steps an image will be used without its caption. 0 means always use captions, while 1 means never use them. Dropping captions helps capture more details of an image, and can prevent over-fitting words with specific image elements. Try higher values when training a style.",
        default=0.05,
        ge=0,
        le=1,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/flux-dev-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
) -> TrainingOutput:
    clean_up()

    run_name = 'training'
    if hf_token is not None:
        login(token=hf_token.get_secret_value())

    print(config_path)
    config = get_config(config_path)

    if lora_rank is not None:
        config['train']["lora_config"]["r"] = lora_rank
        config['train']["lora_config"]["lora_alpha"] = lora_rank
    
    if learning_rate is not None:
        config['train']["optimizer"]['params']["lr"] = learning_rate
      
    config['train']['save_path'] = str(OUTPUT_DIR)
    training_config = config["train"]
      

    dataset = load_dataset(dataset_repo, split="train")

    if training_config["dataset"]["type"] == "cartoon":
        dataset = CartoonDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            padding=training_config["dataset"]["padding"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
        )
    
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )


    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    print(config['train']['lora_config'])

    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    trainer.fit(trainable_model, train_loader)

    output_path = OUTPUT_DIR / run_name
    

    api = HfApi()

    repo_url = api.create_repo(
        hf_repo_id,
        private=True,
        exist_ok=True,
        token=hf_token.get_secret_value(),
    )

    print(f"HF Repo URL: {repo_url}")
    api.upload_folder(
        repo_id=hf_repo_id,
        folder_path=output_path,
        repo_type="model",
        use_auth_token=hf_token.get_secret_value(),
    )

    return TrainingOutput(weights=Path(output_path))






def clean_up():
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


train(
  config_path="train/config/cartoon_1024.yaml",
  lora_rank=1,
)  # Run the training