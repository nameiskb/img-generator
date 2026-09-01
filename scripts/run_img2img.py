"""GitHub Actions 러너 위에서 상품 배경 교체(인페인팅)를 직접 실행한다.

기존엔 img2img(전체 이미지에 균일하게 노이즈)를 썼는데, 강도를 낮추면 배경이 원본
그대로 남고 강도를 높이면 상품 라벨/텍스트까지 뭉개지는 문제가 있었다(2026-08-31,
실제 여러 강도로 테스트해서 확인). "상품은 원본 그대로, 배경만 새로 그리기"를
동시에 만족하려면 균일한 img2img로는 구조적으로 불가능하다 — 그래서 인페인팅으로
바꿨다: rembg로 상품 부분을 분리한 뒤, 그 부분만 마스크로 보호하고 배경만 새로
생성한다. 마스크로 보호된 영역은 파이프라인이 원본 픽셀을 그대로 복원해주므로
상품 라벨/텍스트가 전혀 손상되지 않는다.
"""
from __future__ import annotations

import os
from io import BytesIO

import requests
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter
from rembg import remove

MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"


def _build_masks(original: Image.Image) -> tuple[Image.Image, Image.Image]:
    """rembg로 상품(전경)을 분리해서, 인페인팅용 마스크(배경=흰색=다시 그림,
    상품=검은색=원본 보존)와 합성용 전경 알파를 반환한다."""
    cutout = remove(original)  # RGBA, 상품만 남고 배경은 투명
    alpha = cutout.split()[-1]
    # 경계가 너무 날카로우면 합성 자국이 티나서 살짝 블러 처리
    alpha_soft = alpha.filter(ImageFilter.GaussianBlur(radius=3))
    mask = Image.eval(alpha_soft, lambda a: 255 - a)  # 상품=0(보존), 배경=255(재생성)
    return mask, alpha_soft


def main() -> None:
    prompt = os.environ["PROMPT"]
    image_url = os.environ["IMAGE_URL"]
    strength = float(os.environ.get("STRENGTH", "0.99"))

    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    original = Image.open(BytesIO(resp.content)).convert("RGB").resize((512, 512))

    mask, alpha_soft = _build_masks(original)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cpu")

    generated = pipe(
        prompt=prompt,
        image=original,
        mask_image=mask,
        strength=strength,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    # 파이프라인이 자체적으로 마스크 합성을 하지만, 경계를 한 번 더 부드럽게
    # 다듬기 위해 원본 상품을 알파 블렌딩으로 다시 한번 올려준다.
    final = Image.composite(original, generated, alpha_soft)
    final.save("output.png")
    print("saved output.png")


if __name__ == "__main__":
    main()
