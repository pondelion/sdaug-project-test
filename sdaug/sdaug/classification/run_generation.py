import argparse
import os
from datetime import datetime
from typing import List

from fastprogress import progress_bar as pb

from sdaug.classification.prompt_generator import (
    ClassificationRulebaseAugPromptGenerator,
)
from sdaug.sdwebui_api import SDWebuiAPI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./gen_images/classification")
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
    args = parser.parse_args()
    return args


def run_generation(
    output_dir: str,
    exp_name: str,
    n_gen_per_subject: int,
    sdwebui_url: str,
    subject_list: List[str],
    adjective_list: List[str] = [],
    verb_list: List[str] = [],
    steps: int = 50,
) -> str:
    base_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(base_output_dir, exist_ok=True)

    for subject in pb(subject_list):
        prompt_generator = ClassificationRulebaseAugPromptGenerator(
            subject_list=[subject],
            adjective_list=adjective_list,
            verb_list=verb_list,
        )
        prommpt_list = prompt_generator.gen_random_sentence(n=n_gen_per_subject)
        output_dir = os.path.join(base_output_dir, subject)
        os.makedirs(output_dir, exist_ok=True)
        print(f"generating subject : {subject}")
        for i, prompt in enumerate(pb(prommpt_list)):
            print(f'  generating prompt "{prompt}" ...')
            gen_image = SDWebuiAPI.txt2img(
                sdwebui_url=sdwebui_url,
                prompt=prompt,
                steps=steps,
            )
            output_path = os.path.join(output_dir, f"{i:06d}.jpg")
            gen_image.save(output_path)
    print(f"generated images are saved to {base_output_dir}")
    return base_output_dir


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
