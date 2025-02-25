import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import argparse
import pandas as pd
import json
from datasets import load_dataset
from tqdm import tqdm
from torchvision import transforms


def main(args):
    cuda_number = args.cuda

    if torch.cuda.is_available():
        cur_device = "cuda:" + str(cuda_number)
    else:
        cur_device = "cpu"

    model = "runwayml/stable-diffusion-v1-5" if args.model.upper() == "SD1.5" else "stabilityai/stable-diffusion-2-1"
    pipeline = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16).to(cur_device)
    pipeline.set_progress_bar_config(disable=True)

    with open(args.context_dir, "r") as file:
        eval_datum = json.load(file)["data"]

    for datum in tqdm(eval_datum):
        context = datum["context"]
        id = datum["id"]
        for idx_img in range(4):
            image = pipeline(context, num_inference_steps=args.num_inference_steps).images[0]
            image.save(arg.output_dir + "/{:}_gen_{:}.png".format(id, idx_img))


if __name__ == '__main__':
    # "stabilityai/stable-diffusion-2-1" stable diffusion 2.1

    # Stable Diffusion 1.5
    # runwayml/stable-diffusion-v1-5 : model_name
    # stabilityai/stable-diffusion-1-5 : lora_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--model", type=str, default="SD1.5")
    parser.add_argument("--context_dir", type=str,
                        default="/Dataset/clear_dir_prompt_image_total.json")
    parser.add_argument("--output_dir", type=str,
                        default="/image_gen/clear_stable-diffusion-1-5")
    parser.add_argument("--num_repeat", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=80)
    arg = parser.parse_args()
    main(arg)
