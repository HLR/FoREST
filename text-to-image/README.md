# Text-to-Image Generation (T2I)

# **Under development**

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

- --image_dir
- --text_file
- --grounding_dino_config_path
- --grounding_dino_weights_path
- --output_result_dir
- --text_result_dir
- --device


**The evaluation results will be saved in file provided in --output_result_dir as csv format and --text_result_dir as text format.** 

---