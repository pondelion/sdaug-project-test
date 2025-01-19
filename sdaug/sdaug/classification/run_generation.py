import argparse
import base64
from datetime import datetime
import os
import io
from typing import Optional

import requests
from fastprogress import progress_bar as pb
from PIL import Image

from sdaug.classification.prompt_generator import ClassificationRulebaseAugPromptGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./gen_images/classification")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument("--sdwebui_url", type=str, default="http://127.0.0.1:7860")
    parser.add_argument("--n_gen_per_subject", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument('--subject_list', required=True, nargs="+", type=str)
    parser.add_argument('--adjective_list', required=False, nargs="?", type=str, default=[])
    parser.add_argument('--verb_list', required=False, nargs="?", type=str, default=[])
    args = parser.parse_args()
    return args


def call_txt2img_api(
    sdwebui_url: str,
    prompt: str,
    steps: Optional[int] = None,
) -> Image.Image:
    payload = {
        'prompt': prompt,
    }
    if steps is not None:
        payload['steps'] = steps

    response = requests.post(url=os.path.join(sdwebui_url, 'sdapi/v1/txt2img'), json=payload)
    r = response.json()

    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    return image


def run_generation():
    args = parse_args()
    print(args)
    base_output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(base_output_dir, exist_ok=True)

    for subject in  pb(args.subject_list):
        prompt_generator = ClassificationRulebaseAugPromptGenerator(
            subject_list=[subject],
            adjective_list=args.adjective_list,
            verb_list=args.verb_list,
        )
        prommpt_list = prompt_generator.gen_random_sentence(n=args.n_gen_per_subject)
        output_dir = os.path.join(base_output_dir, subject)
        os.makedirs(output_dir, exist_ok=True)
        print(f'generating subject : {subject}')
        for i, prompt in  enumerate(pb(prommpt_list)):
            print(f'  generating prompt "{prompt}" ...')
            gen_image = call_txt2img_api(
                sdwebui_url=args.sdwebui_url,
                prompt=prompt,
                steps=args.steps,
            )
            output_path = os.path.join(output_dir, f'{i:06d}.jpg')
            gen_image.save(output_path)
    print(f'generated images are saved to {base_output_dir}')


if __name__ == "__main__":
    run_generation()
