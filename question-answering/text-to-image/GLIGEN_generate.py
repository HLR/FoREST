import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image
import argparse
import pandas as pd
import json
from tqdm import tqdm
import os


def main(arg):
    from accelerate.utils import set_seed
    set_seed(0)

    cuda_number = arg.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        if torch.cuda.is_available():
            cur_device = "cuda:" + str(cuda_number)
        else:
            cur_device = "cpu"

    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)

    # Generate an image described by the prompt and
    # insert objects described by text at the region defined by bounding boxes
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to(cur_device)
    pipe.set_progress_bar_config(disable=True)

    layout_data: pd.DataFrame = pd.read_csv(arg.layout_dir, index_col=0)

    map_id_layout = {}
    for idx, data in layout_data.iterrows():
        id_split = data["id"].split("_")
        if id_split[-1].isdigit():
            id_split = id_split[:-1]
        image_id = "_".join(id_split)
        if data["layout"].find("Objects: ") == -1:
            map_id_layout[image_id] = []
        else:
            try:
                layout = data["layout"][data["layout"].find("Objects: ") + 9:data["layout"].rfind("]") + 1]
                map_id_layout[image_id] = eval(layout)
            except:
                map_id_layout[image_id] = []

    with open(arg.context_dir, "r") as file:
        eval_datum = json.load(file)["data"]

    for eval_data in tqdm(eval_datum):
        context = eval_data["context"]
        id_data = eval_data["id"]
        context_id = f"{id_data}"
        layouts = map_id_layout[context_id]
        reference_obj = eval_data["obj2"]
        reference_obj_dir = eval_data["obj2_dir"]
        boxes = []
        phrases = []
        if layouts:  # If layout generated correctly, use the layout information
            for layout in layouts:
                # print(layout)
                # layout = eval(layout)
                phrase = layout[0]
                box = layout[1]
                if phrase == "my perspective":
                    continue
                if len(box) != 4:
                    continue
                # Adding orientation phase to relatum
                if arg.direction and phrase == reference_obj:
                    if reference_obj_dir == "front":
                        phrase = phrase + " that is facing toward camera."
                    elif reference_obj_dir == "left":
                        phrase = phrase + " that is facing to the left."
                    elif reference_obj_dir == "back":
                        phrase = phrase + " that is facing away from the camera."
                    elif reference_obj_dir == "right":
                        phrase = phrase + " that is facing to the right."

                xmin = box[0]
                ymin = box[1]
                xmax = xmin + box[2]
                ymax = ymin + box[3]
                if not (isinstance(xmin, int) and isinstance(xmax, int) and isinstance(ymin, int) and isinstance(ymax,
                                                                                                                 int)):
                    continue
                box = [xmin / 512, ymin / 512, xmax / 512, ymax / 512]
                boxes.append(box)
                phrases.append(phrase)

        for gen_img in range(arg.num_repeat):
            images = pipe(prompt=context,
                          gligen_phrases=phrases if phrases else [""],
                          gligen_boxes=boxes if boxes else [[0, 0, 0, 0]],
                          output_type="pil",
                          gligen_scheduled_sampling_beta=0.3,
                          num_inference_steps=50
                          ).images

            images[0].save(arg.output_dir + "/{:}_gen_{:}.png".format(context_id, gen_img))


if __name__ == '__main__':
    # llama_layout_v2_4-shot_without_dir.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=7)
    parser.add_argument("--layout_dir", type=str,
                        default="/LLMs_results/llama_layout_v2_4-shot.csv")
    parser.add_argument("--context_dir", type=str,
                        default="/Dataset/C-split_image_total.json")
    parser.add_argument("--output_dir", type=str,
                        default="/image_gen/GLIGEN/image_gen_layout")
    parser.add_argument("--num_repeat", type=int, default=4)
    parser.add_argument("--direction", type=bool, default=True)

    arg = parser.parse_args()
    main(arg)
