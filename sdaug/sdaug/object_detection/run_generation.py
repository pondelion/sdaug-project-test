import argparse
from datetime import datetime
from typing import Any, Dict

from sdaug.classification.run_generation import run_generation as run_txt2img_generation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./gen_images/object_detection"
    )
    parser.add_argument(
        "--exp_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    parser.add_argument("--sdwebui_url", type=str, default="http://127.0.0.1:7860")
    parser.add_argument("--n_gen_per_subject", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--subject_list", required=True, nargs="+", type=str)
    parser.add_argument(
        "--adjective_list", required=False, nargs="?", type=str, default=[]
    )
    parser.add_argument("--verb_list", required=False, nargs="?", type=str, default=[])
    parser.add_argument("--anno_format", choices=["coco"], default="coco")
    args = parser.parse_args()
    return args


def run_generation(
    txt2im_gen_kwargs: Dict[str, Any],
):
    # step 1) generate each subject images
    gen_subject_img_dir = run_txt2img_generation(**txt2im_gen_kwargs)
    # step 2) do background removal and create image with alpha background
    # step 3) create augumentated object detection annotation dataset by combining alpha subject image and background inpainting


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_generation(
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        n_gen_per_subject=args.n_gen_per_subject,
        sdwebui_url=args.sdwebui_url,
        subject_list=args.subject_list,
        adjective_list=args.adjective_list,
        verb_list=args.verb_list,
        steps=50,
    )
