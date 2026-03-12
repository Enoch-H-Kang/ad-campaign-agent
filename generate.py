"""Image generation backends."""

import base64
import tempfile
from pathlib import Path

from openai import OpenAI


# ─── OpenAI Image Generation (default, everyone has access) ────

def generate_image_openai(
    client: OpenAI,
    prompt: str,
    size: str = "1536x1024",
    quality: str = "high",
    model: str = "gpt-image-1.5",
) -> tuple[Path | None, str | None]:
    """Generate an ad campaign image using OpenAI's image API.

    Returns:
        (path, None) on success, (None, error_message) on failure.
    """
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
            output_format="png",
        )

        image_data = response.data[0].b64_json
        if not image_data:
            return None, "No image data in OpenAI response."

        output_path = Path(tempfile.mkdtemp()) / "ad_campaign.png"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_data))

        return output_path, None
    except Exception as e:
        return None, f"OpenAI error: {e}"


# ─── Gemini Native Image (style-reference-aware) ────────────────

def generate_image_gemini(
    api_key: str,
    prompt: str,
    style_image_bytes: bytes | None = None,
    model: str = "gemini-2.5-flash-image",
) -> tuple[Path | None, str | None]:
    """Generate an ad image using Gemini native image generation.

    Accepts a style reference image to guide visual direction and aesthetic.

    Returns:
        (path, None) on success, (None, error_message) on failure.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        contents = []
        if style_image_bytes:
            contents.append(types.Part.from_bytes(
                data=style_image_bytes, mime_type="image/png",
            ))
        contents.append(prompt)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                output_path = Path(tempfile.mkdtemp()) / "ad_campaign.png"
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                return output_path, None

        # No image in response — check for text feedback
        text_parts = [
            p.text for p in response.candidates[0].content.parts
            if hasattr(p, "text") and p.text
        ]
        return None, f"No image generated. Model response: {'; '.join(text_parts) if text_parts else 'empty response'}"
    except Exception as e:
        return None, f"Gemini error: {e}"


# ─── Google Imagen 4 (requires Gemini API key) ──────────────────

def generate_image_imagen(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    model: str = "imagen-4.0-fast-generate-001",
) -> tuple[Path | None, str | None]:
    """Generate an ad campaign image using Google Imagen 4.

    Returns:
        (path, None) on success, (None, error_message) on failure.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                output_mime_type="image/png",
            ),
        )

        if not response.generated_images:
            return None, "Imagen returned no images. The prompt may have been filtered."

        output_path = Path(tempfile.mkdtemp()) / "ad_campaign.png"
        response.generated_images[0].image.save(str(output_path))
        return output_path, None
    except Exception as e:
        return None, f"Imagen error: {e}"
