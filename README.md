# OminiControl


<img src='./assets/demo/demo_this_is_omini_control.jpg' width='100%' />
<br>

<a href="https://huggingface.co/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FYuanshi%2FOminiControl%3Fexpand%255B%255D%3DdownloadsAllTime&amp;search=%22downloadsAllTime%22%3A%5Cs%2A%28%5Cd%2B%29%28%5Cd%29%5Cd%7B2%7D&amp;replace=Model%20%28%241.%242k%20downloads%29&amp;label=%F0%9F%A4%97%20HuggingFace&amp;color=ffbd45" alt="HuggingFace Model Downloads" title="All-time model downloads on Hugging Face"></a>
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Demo-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl_Art"><img src="https://img.shields.io/badge/🤗_HuggingFace-Demo2-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://github.com/Yuanshi9815/Subjects200K"><img src="https://img.shields.io/badge/GitHub-Dataset-blue.svg?logo=github&" alt="GitHub"></a>
<a href="https://huggingface.co/datasets/Yuanshi/Subjects200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>
<br>
<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/arXiv-OminiControl-A42C25.svg" alt="arXiv"></a>
<a href="https://arxiv.org/abs/2503.08280"><img src="https://img.shields.io/badge/arXiv-OminiControl2-A42C25.svg" alt="arXiv"></a>

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

> **OminiControl2: Efficient Conditioning for Diffusion Transformers**
> <br>
> Zhenxiong Tan, 
> Qiaochu Xue, 
> [Xingyi Yang](https://adamdad.github.io/), 
> [Songhua Liu](http://121.37.94.87/), 
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
- **2026-07-02**: Codebase updated for `diffusers` 0.38. `generate()` gains a `condition_scale` argument to adjust condition strength, subject-driven generation now works on `FLUX.1-dev` (see [example](./examples/subject_dev.ipynb)), and OminiControl2's KV-cache fast inference (`kv_cache=True`) is now documented (see [Usage example](#usage-example)).
- **2025-05-12**: ⭐️ The code of [OminiControl2](https://arxiv.org/abs/2503.08280) is released. It introduces a new efficient conditioning method for diffusion transformers. (Check out the training code [here](./train)).
- **2025-05-12**: Support custom style LoRA. (Check out the [example](./examples/combine_with_style_lora.ipynb)).
- **2025-04-09**: ⭐️ [OminiControl Art](https://huggingface.co/spaces/Yuanshi/OminiControl_Art) is released. It can stylize any image with an artistic style. (Check out the [demo](https://huggingface.co/spaces/Yuanshi/OminiControl_Art) and [inference examples](./examples/ominicontrol_art.ipynb)).
- **2024-12-26**: Training code is released. Now you can create your own OminiControl model by customizing any control tasks (3D, multi-view, pose-guided, try-on, etc.) with the FLUX model. Check the [training folder](./train) for more details.

## Quick Start
### Setup (Optional)
1. **Environment setup**
```bash
conda create -n omini python=3.12
conda activate omini
```
2. **Torch installation** (pick the index URL matching your CUDA version)
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```
3. **Requirements installation**
```bash
pip install -r requirements.txt
```
### Usage example
1. Subject-driven generation: `examples/subject.ipynb` (on `FLUX.1-dev`: `examples/subject_dev.ipynb`)
2. Subject-driven generation at 1024x1024: `examples/subject_1024.ipynb` (best quality with the 1024-trained model)
3. In-painting: `examples/inpainting.ipynb`
4. Canny edge to image, depth to image, colorization, deblurring: `examples/spatial.ipynb`
5. Combining with a custom style LoRA: `examples/combine_with_style_lora.ipynb`
6. Image stylization (OminiControl Art): `examples/ominicontrol_art.ipynb`

> **Note (multiple LoRAs):** if you load more than one adapter via repeated `pipe.load_lora_weights(..., adapter_name=...)` calls, activate them explicitly with `pipe.set_adapters(["adapter_a", "adapter_b"])` — otherwise only the last-loaded adapter stays active and the others are silently ignored. See `examples/spatial.ipynb` for the pattern.

#### Adjusting condition strength
`generate()` accepts a `condition_scale` argument (default `1.0`, which reproduces the original behavior exactly). Values `> 1` strengthen the condition image's influence, values `< 1` weaken it, and `0` suppresses it entirely.
```python
result = generate(pipe, prompt=prompt, conditions=[condition], condition_scale=1.3)
```

#### Faster generation with KV-cache (OminiControl2)
Pass `kv_cache=True` to `generate()` to compute the condition branch's keys/values once and reuse them across all remaining steps (~1.5x end-to-end speedup at 8 steps). This requires a LoRA trained with `independent_condition: true` — see [Efficient Generation (OminiControl2)](./train/README.md#efficient-generation-ominicontrol2).

### Guidelines for subject-driven generation
1. Center-crop and resize your input image to the model's native resolution (512x512 for `subject`/`subject_512`, 1024x1024 for `subject_1024_beta`) before passing it in — the pipeline does not do this automatically. See the example notebooks for the preprocessing code.
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
- Prompt4: *On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.*
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
   
### Stylize images
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl_Art"><img src="https://img.shields.io/badge/🤗_HuggingFace-Demo2-ffbd45.svg" alt="HuggingFace"></a>
</br>
<img src='./assets/demo/art1.png' width='600px' />
<img src='./assets/demo/art2.png' width='600px' />
</br>



## Models

**Subject-driven control:**
| Model                                                                                            | Description                                                                                                                                                                | Resolution   |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| [`experimental`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) / `subject` | The model used in the paper.                                                                                                                                               | (512, 512)   |
| [`omini`](https://huggingface.co/Yuanshi/OminiControl/tree/main/omini) / `subject_512`           | The model has been fine-tuned on a larger dataset.                                                                                                                         | (512, 512)   |
| [`omini`](https://huggingface.co/Yuanshi/OminiControl/tree/main/omini) / `subject_1024_beta`     | The model has been fine-tuned on a larger dataset and trained at 1024x1024.                                                                                                | (1024, 1024) |
| [`oye-cartoon`](https://huggingface.co/saquiboye/oye-cartoon)                                    | Fine-tuned on the [oye-cartoon](https://huggingface.co/datasets/saquiboye/oye-cartoon) dataset by [@saquib764](https://github.com/Saquib764) (for `FLUX.1-dev`)            | (512, 512)   |

> The subject LoRAs were trained on `FLUX.1-dev`. When running them on `FLUX.1-dev`, enable real image guidance (`image_guidance_scale > 1.0`, keep `guidance_scale=3.5`) — see `examples/subject_dev.ipynb`. They also run on `FLUX.1-schnell` (as in the example notebooks), where no image guidance is needed.
>
> `subject_1024_beta` was trained at 1024x1024 and gives its best results at that resolution. The weight currently lives in a non-main revision of the HF repo — pass `revision=` when loading it (see `examples/subject_1024.ipynb`).

**Spatially aligned control:**
| Model                                                                                                     | Description                                                                | Resolution   |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------ |
| [`experimental`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) / `<task_name>`      | Canny edge to image (`canny`), depth to image (`depth`), colorization (`coloring`), deblurring (`deblurring`), in-painting (`fill`). Works on both `FLUX.1-dev` and `FLUX.1-schnell`. | (512, 512)   |

## Community Extensions
- [ComfyUI-Diffusers-OminiControl](https://github.com/Macoron/ComfyUI-Diffusers-OminiControl) - ComfyUI integration by [@Macoron](https://github.com/Macoron)
- [ComfyUI_RH_OminiControl](https://github.com/HM-RunningHub/ComfyUI_RH_OminiControl) - ComfyUI integration by [@HM-RunningHub](https://github.com/HM-RunningHub)

## Limitations
1. The model's subject-driven generation primarily works with objects rather than human subjects due to the absence of human data in training.
2. The subject-driven LoRAs were trained on `FLUX.1-dev`. When running them on `FLUX.1-dev`, you must use real image guidance: call `generate(...)` with `image_guidance_scale > 1.0` (e.g. `1.5`) and more steps (~20–28) — without it, `FLUX.1-dev` tends to ignore the condition. `image_guidance_scale` is the tunable CFG knob; the distilled `guidance_scale` must be kept at `3.5` (the value used in training, for train/inference consistency — it is not a free hyperparameter). See `examples/subject_dev.ipynb`. On `FLUX.1-schnell` (as in the example notebooks), no image guidance is needed.
3. The `subject`/`subject_512` and spatial LoRAs were trained at 512x512 and work best at that resolution; `subject_1024_beta` was trained at 1024x1024 and gives its best results there (see `examples/subject_1024.ipynb`). Non-square resolutions (e.g. 512x768) also work.

## Training
Training instructions can be found in this [folder](./train).


## To-do
- [x] Release the training code.
- [x] Release the model for higher resolution (1024x1024).

## Acknowledgment
We would like to acknowledge that the computational work involved in this research work is partially supported by NUS IT’s Research Computing group using grant numbers NUSREC-HPC-00001.

## Citation
```
@inproceedings{tan2025ominicontrol,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Tan, Zhenxiong and Liu, Songhua and Yang, Xingyi and Xue, Qiaochu and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}

@inproceedings{tan2026ominicontrol2,
  title={Ominicontrol2: Efficient conditioning for diffusion transformers},
  author={Tan, Zhenxiong and Xue, Qiaochu and Yang, Xingyi and Liu, Songhua and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4256--4265},
  year={2026}
}
```
