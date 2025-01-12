# OminiControl Training 🛠️

## Preparation

### Setup
1. **Environment**
    ```bash
    conda create -n omini python=3.10
    conda activate omini
    ```
2. **Requirements**
    ```bash
    pip install -r train/requirements.txt
    ```

### Dataset
1. Download dataset [Subject200K](https://huggingface.co/datasets/Yuanshi/Subjects200K). (**subject-driven generation**)
    ```
    bash train/script/data_download/data_download1.sh
    ```
2. Download dataset [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M). (**spatial control task**)
    ```
    bash train/script/data_download/data_download2.sh
    ```
    **Note:** By default, only a few files are downloaded. You can modify `data_download2.sh` to download additional datasets. Remember to update the config file to specify the training data accordingly.

## Training

### Start training training
**Config file path**: `./train/config`

**Scripts path**: `./train/script`

1. Subject-driven generation
    ```bash
    bash train/script/train_subject.sh
    ```
2. Spatial control task
    ```bash
    bash train/script/train_canny.sh
    ```

**Note**: Detailed WanDB settings and GPU settings can be found in the script files and the config files.

### Other spatial control tasks
This repository supports 5 spatial control tasks: 
1. Canny edge to image (`canny`)
2. Image colorization (`coloring`)
3. Image deblurring (`deblurring`)
4. Depth map to image (`depth`)
5. Image to depth map  (`depth_pred`)
6. Image inpainting (`fill`)
7. Super resolution (`sr`)

You can modify the `condition_type` parameter in config file `config/canny_512.yaml` to switch between different tasks.

### Customize your own task
You can customize your own task by constructing a new dataset and modifying the training code.

<details>
<summary>Instructions</summary>

1. **Dataset** : 
   
   Construct a new dataset with the following format: (`src/train/data.py`)
    ```python
    class MyDataset(Dataset):
        def __init__(self, ...):
            ...
        def __len__(self):
            ...
        def __getitem__(self, idx):
            ...
            return {
                "image": image,
                "condition": condition_img,
                "condition_type": "your_condition_type",
                "description": description,
                "position_delta": position_delta
            }
    ```
    **Note:** For spatial control tasks, set the `position_delta` to be `[0, 0]`. For non-spatial control tasks, set `position_delta` to be `[0, -condition_width // 16]`.
2. **Condition**:
   
   Add a new condition type in the `Condition` class. (`src/flux/condition.py`)
    ```python
    condition_dict = {
        ...
        "your_condition_type": your_condition_id_number, # Add your condition type here
    }
    ...
    if condition_type in [
        ...
        "your_condition_type", # Add your condition type here
    ]:
        ...
    ```
3. **Test**: 
   
   Add a new test function for your task. (`src/train/callbacks.py`)
    ```python
    if self.condition_type == "your_condition_type":
        condition_img = (
            Image.open("images/vase.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        ...
        test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
    ```

4. **Import relevant dataset in the training script**
   Update the file in the following section. (`src/train/train.py`)
   ```python
    from .data import (
        ImageConditionDataset,
        Subject200KDateset,
        MyDataset
    )
    ...
   
    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "your_condition_type":
       ...
   ```
   
</details>

## Hardware requirement
**Note**: Memory optimization (like dynamic T5 model loading) is pending implementation.
- Hardware: 2x NVIDIA H100 GPUs
- Memory: ~80GB GPU memory
