# OminiControl


<img src='./assets/demo/demo_this_is_omini_control.jpg' width='100%' />
<br>

<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/ariXv-2411.15098-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://github.com/Yuanshi9815/Subjects200K"><img src="https://img.shields.io/badge/GitHub-Dataset-blue.svg?logo=github&" alt="GitHub"></a>
<a href="https://huggingface.co/datasets/Yuanshi/Subjects200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

> **OminiControl: Minimal and Universal Control for Diffusion Transformer**
> <br>
> Zhenxiong Tan, 
> [Songhua Liu](http://121.37.94.87/), 
> [Xingyi Yang](https://adamdad.github.io/), 
> Qiaochu Xue, 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> [xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore
> <br>


## Features

OminiControl is a minimal yet powerful universal control framework for Diffusion Transformer models like [FLUX](https://github.com/black-forest-labs/flux).

* **Universal Control 🌐**:  A unified control framework that supports both subject-driven control and spatial control (such as edge-guided and in-painting generation).

* **Minimal Design 🚀**: Injects control signals while preserving original model structure. Only introduces 0.1% additional parameters to the base model.

## News
- **2024-12-26**: ⭐️ Training code are released. Now you can create your own OminiControl model by customizing any control tasks (3D, multi-view, pose-guided, try-on, etc.) with the FLUX model. Check the [training folder](./train) for more details.

## Quick Start
### Setup (Optional)
1. **Environment setup**
```bash
conda create -n omini python=3.10
conda activate omini
```
2. **Requirements installation**
```bash
pip install -r requirements.txt
```
### Usage example
1. Subject-driven generation: `examples/subject.ipynb`
2. In-painting: `examples/inpainting.ipynb`
3. Canny edge to image, depth to image, colorization, deblurring: `examples/spatial.ipynb`

### Gradio app
To run the Gradio app for subject-driven generation:
```bash
python -m src.gradio.gradio_app
```

### Guidelines for subject-driven generation
1. Input images are automatically center-cropped and resized to 512x512 resolution.
2. When writing prompts, refer to the subject using phrases like `this item`, `the object`, or `it`. e.g.
   1. *A close up view of this item. It is placed on a wooden table.*
   2. *A young lady is wearing this shirt.*
3. The model primarily works with objects rather than human subjects currently, due to the absence of human data in training.

## Generated samples
### Subject-driven generation
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>

**Demos** (Left: condition image; Right: generated image)

<div float="left">
  <img src='./assets/demo/oranges_omini.jpg' width='48%'/>
  <img src='./assets/demo/rc_car_omini.jpg' width='48%' />
  <img src='./assets/demo/clock_omini.jpg' width='48%' />
  <img src='./assets/demo/shirt_omini.jpg' width='48%' />
</div>

<details>
<summary>Text Prompts</summary>

- Prompt1: *A close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!.'*
- Prompt2: *A film style shot. On the moon, this item drives across the moon surface. A flag on it reads 'Omini'. The background is that Earth looms large in the foreground.*
- Prompt3: *In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.*
- Prompt4: *"On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple."*
</details>
<details>
<summary>More results</summary>

* Try on:
  <img src='./assets/demo/try_on.jpg'/>
* Scene variations:
  <img src='./assets/demo/scene_variation.jpg'/>
* Dreambooth dataset:
  <img src='./assets/demo/dreambooth_res.jpg'/>
* Oye-cartoon finetune:
  <div float="left">
    <img src='./assets/demo/man_omini.jpg' width='48%' />
    <img src='./assets/demo/panda_omini.jpg' width='48%' />
  </div>
</details>

### Spatially aligned control
1. **Image Inpainting** (Left: original image; Center: masked image; Right: filled image)
  - Prompt: *The Mona Lisa is wearing a white VR headset with 'Omini' written on it.*
    </br>
    <img src='./assets/demo/monalisa_omini.jpg' width='700px' />
  - Prompt: *A yellow book with the word 'OMINI' in large font on the cover. The text 'for FLUX' appears at the bottom.*
    </br>
    <img src='./assets/demo/book_omini.jpg' width='700px' />
2. **Other spatially aligned tasks**  (Canny edge to image, depth to image, colorization, deblurring) 
    </br>
    <details>
    <summary>Click to show</summary>
    <div float="left">
      <img src='./assets/demo/room_corner_canny.jpg' width='48%'/>
      <img src='./assets/demo/room_corner_depth.jpg' width='48%' />
      <img src='./assets/demo/room_corner_coloring.jpg' width='48%' />
      <img src='./assets/demo/room_corner_deblurring.jpg' width='48%' />
    </div>
    
    Prompt: *A light gray sofa stands against a white wall, featuring a black and white geometric patterned pillow. A white side table sits next to the sofa, topped with a white adjustable desk lamp and some books. Dark hardwood flooring contrasts with the pale walls and furniture.*
    </details>
   



## Models

**Subject-driven control:**
| Model                                                                                            | Base model     | Description                                                                                              | Resolution   |
| ------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------------------------------- | ------------ |
| [`experimental`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) / `subject` | FLUX.1-schnell | The model used in the paper.                                                                             | (512, 512)   |
| [`omini`](https://huggingface.co/Yuanshi/OminiControl/tree/main/omini) / `subject_512`           | FLUX.1-schnell | The model has been fine-tuned on a larger dataset.                                                       | (512, 512)   |
| [`omini`](https://huggingface.co/Yuanshi/OminiControl/tree/main/omini) / `subject_1024`          | FLUX.1-schnell | The model has been fine-tuned on a larger dataset and accommodates higher resolution.   (To be released) | (1024, 1024) |
| [`oye-cartoon`](https://huggingface.co/saquiboye/oye-cartoon)          | FLUX.1-dev | The model has been fine-tuned on [oye-cartoon](https://huggingface.co/datasets/saquiboye/oye-cartoon) dataset by [@saquib764](https://github.com/Saquib764) | (512, 512) |

**Spatial aligned control:**
| Model                                                                                                     | Base model | Description                                                                | Resolution   |
| --------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------- | ------------ |
| [`experimental`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) / `<task_name>`      | FLUX.1     | Canny edge to image, depth to image, colorization, deblurring, in-painting | (512, 512)   |
| [`experimental`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) / `<task_name>_1024` | FLUX.1     | Supports higher resolution.(To be released)                                | (1024, 1024) |

## Community Extensions
- [ComfyUI-Diffusers-OminiControl](https://github.com/Macoron/ComfyUI-Diffusers-OminiControl) - ComfyUI integration by [@Macoron](https://github.com/Macoron)
- [ComfyUI_RH_OminiControl](https://github.com/HM-RunningHub/ComfyUI_RH_OminiControl) - ComfyUI integration by [@HM-RunningHub](https://github.com/HM-RunningHub)

## Limitations
1. The model's subject-driven generation primarily works with objects rather than human subjects due to the absence of human data in training.
2. The subject-driven generation model may not work well with `FLUX.1-dev`.
3. The released model currently only supports the resolution of 512x512.

## Training
Training instructions can be found in this [folder](./train).


## To-do
- [x] Release the training code.
- [ ] Release the model for higher resolution (1024x1024).

## Citation
```
@article{tan2024ominicontrol,
  title={Ominicontrol: Minimal and universal control for diffusion transformer},
  author={Tan, Zhenxiong and Liu, Songhua and Yang, Xingyi and Xue, Qiaochu and Wang, Xinchao},
  journal={arXiv preprint arXiv:2411.15098},
  volume={3},
  year={2024}
}
```
