"""GitHub Actions 러너 위에서 Stable Diffusion img2img를 직접 실행한다.

thread-automation 프로젝트의 오라클 서버가 repository_dispatch로 이 워크플로를
트리거하면, 여기서 실제 모델을 CPU로 돌려서 output.png를 만든다. 무료 img2img API가
실제로는 존재하지 않아서(Cloudflare/Pollinations/HuggingFace 전부 확인됨) 대신
오픈소스 모델 자체를 우리가 직접 실행하는 방식으로 우회한다.
"""
from __future__ import annotations

import os
from io import BytesIO

import requests
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main() -> None:
    prompt = os.environ["PROMPT"]
    image_url = os.environ["IMAGE_URL"]
    strength = float(os.environ.get("STRENGTH", "0.5"))

    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    init_image = Image.open(BytesIO(resp.content)).convert("RGB").resize((512, 512))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cpu")

    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    result.save("output.png")
    print("saved output.png")


if __name__ == "__main__":
    main()
