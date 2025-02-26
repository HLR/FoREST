# Text-to-Image Generation (T2I)

## Task Description

This task aims to determine the diffusion models' ability to consider FoR by evaluating their generated images. 
The input is a spatial expression and the output is a generated image. 
We use the context from both C and A splits with external FoRs for this task.


## Generate Images

### Stable Diffusion

In this model, the model only need context as the input.

#### Command To Run

```bash
python SD_inference.py --cuda 0 --model SD2.1
```

All option parameters are listed below,
- --cuda : GPU number to use to run this
- --model : Either "SD1.5" (Stable-Diffusion 1.5) or "SD2.1" (Stable-Diffusion 2.1)
- --context_dir : file name of the context dataset
- --output_dir : output directory of generated image
- --num_repeat : number of repeated generation per context
- --num_inference_steps : number of inference steps per image

### Layout Diffusion

This model consists of two steps: text-to-layout and layout-to-image

1. Generate the layout using following command (text-to-layout). We only use Llama3 to generate the layout; however, other LLMs should capable of this as well.

```bash
python layout_generation.py
```

All option parameters are listed below,
- --cuda: GPT number to be used
- --model_size: Llama3.0 parameters size to be used (8B / 70 B)
- --few_shot: whether to give the few shot of generating layout in [few_shot_layout](../few_shot_layout) (4, 8, 16)
- --clear: whether to use C-split or A-split
- --SG_information : ignore this option if not need SG information as augmented information. Please run [FoR identification](../FoR-Identification) with SG method first.

2. Generate the image based on generated layout. 

```bash
python GLIGEN_generate.py
```

All option parameters are listed below,
- --cuda: GPT number to be used
- --layout_dir: directory of generated bounding box layouts
- --context_dir: Dataset directory
- --output_dir : output directory of generated image
- --num_repeat : number of repeated generation per context


---
## T2I Evaluation
The source code of VPEval are borrowed from [VPEVAL](https://github.com/aszala/VPEval.git).
Please also read readme from their repository and download from their repository.

**Notes**:

- weight of grounding dino is **groundingdino_swint_ogc.pth**
- the function is adapted from spatialEval of VPEVAL

This function evaluates the generated images compared to spatial relation in the ground-truth.

Example of running this code is

```bash
python evaluate_FoR_identification.py --text_file /Dataset/A-split_QA_camera_perspective.json --image_dir /models/GLIGEN/image_gen_4_shots_dir
```

All options for parameters are described below

- --image_dir : directory of images to be evaluated
- --text_file : context dataset
- --grounding_dino_config_path : grounding dino config path (no need to change this)
- --grounding_dino_weights_path : grounding dino weight path (no need to change this)
- --output_result_dir : Output result file name
- --text_result_dir : Textual result file name
- --device : cuda device to run evaluation


**The evaluation results will be saved in file provided in --output_result_dir as csv format and --text_result_dir as text format.** 

---