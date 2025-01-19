import base64
import io
import os
from typing import Optional

import requests
from PIL import Image, ImageOps


class SDWebuiAPI:
    @staticmethod
    def txt2img(
        sdwebui_url: str,
        prompt: str,
        steps: Optional[int] = None,
    ) -> Image.Image:
        payload = {
            "prompt": prompt,
        }
        if steps is not None:
            payload["steps"] = steps  # type: ignore

        response = requests.post(
            url=os.path.join(sdwebui_url, "sdapi/v1/txt2img"), json=payload
        )
        r = response.json()

        image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))
        return image

    @staticmethod
    def rembg(
        sdwebui_url: str,
        image: Image.Image,
        model: str = "u2net",
    ) -> Image.Image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        payload = {
            "input_image": f"{img_str}",
            "model": model,
            "return_mask": False,
            "alpha_matting": False,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
        }

        response = requests.post(url=os.path.join(sdwebui_url, "rembg"), json=payload)
        r = response.json()

        image = Image.open(io.BytesIO(base64.b64decode(r["image"])))
        return image

    @staticmethod
    def img2img_inpaint(
        sdwebui_url: str,
        image_rgba: Image.Image,
        prompt: str = "",
        steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Image.Image:
        image_rgb = image_rgba.convert("RGB")
        mask = image_rgba.split()[-1]
        mask = ImageOps.invert(mask)

        buffered = io.BytesIO()
        image_rgb.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        mask_str = base64.b64encode(buffered.getvalue()).decode()
        payload = {
            "prompt": prompt,
            "init_images": [img_str],
            "mask": mask_str,
        }
        if steps is not None:
            payload["steps"] = steps  # type: ignore
        if width is not None:
            payload["width"] = width  # type: ignore
        if height is not None:
            payload["height"] = height  # type: ignore

        response = requests.post(
            url=os.path.join(sdwebui_url, "sdapi/v1/img2img"), json=payload
        )
        r = response.json()

        image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))
        return image
