"""Fields schema and validation for the Ad Campaign Agent."""

from pathlib import Path

REQUIRED_FIELDS = {
    "product_name": "Product / Service Name",
    "target_audience": "Target Audience",
    "campaign_goal": "Campaign Goal",
    "key_message": "Key Message / CTA",
    "brand_tone": "Brand Tone",
    "style_reference": "Style Reference",
}

CAMPAIGN_GOALS = ["awareness", "consideration", "conversion", "launch"]

STYLE_PRESETS = {
    "minimalist": {
        "label": "Minimalist",
        "description": (
            "Clean, airy composition with generous white space, soft neutral tones, "
            "simple geometric forms, and understated typography. Light feels natural and diffused."
        ),
    },
    "dark_dramatic": {
        "label": "Dark & Dramatic",
        "description": (
            "Deep blacks and rich shadows with selective high-contrast lighting. "
            "Moody, cinematic atmosphere with bold rim lighting and dark backgrounds."
        ),
    },
    "vibrant_bold": {
        "label": "Vibrant & Bold",
        "description": (
            "Saturated, punchy colors with high energy. Graphic pop-art influences, "
            "strong color blocking, and dynamic composition that demands attention."
        ),
    },
    "lifestyle": {
        "label": "Lifestyle",
        "description": (
            "Warm, natural photography feel with golden-hour lighting. Candid and aspirational, "
            "showing the product in real-world use. Soft bokeh backgrounds."
        ),
    },
    "retro_vintage": {
        "label": "Retro / Vintage",
        "description": (
            "Film-grain texture, muted warm palette with faded highlights. "
            "Nostalgic 70s-80s aesthetic with analog photography qualities and serif typography."
        ),
    },
    "futuristic": {
        "label": "Futuristic",
        "description": (
            "Sleek, tech-forward aesthetic with neon accents, holographic effects, "
            "and dark metallic surfaces. Cool blue-purple palette with glowing light trails."
        ),
    },
}

STYLE_DIR = Path(__file__).parent / "style"

OPTIONAL_FIELDS = {
    "brand_colors": "Brand Colors (hex codes)",
    "logo": "Logo image",
    "competitor_refs": "Competitor References",
    "tagline": "Existing Tagline / Slogan",
    "do_not_include": "Do-Not-Include elements",
}


def validate_fields(session_data: dict) -> list[str]:
    """Return list of missing required field names."""
    missing = []
    for key, label in REQUIRED_FIELDS.items():
        val = session_data.get(key)
        if val is None or val == "" or val == []:
            missing.append(label)
    return missing


def format_fields_for_prompt(session_data: dict) -> str:
    """Format collected fields as text for the LLM."""
    lines = []
    for key, label in {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}.items():
        val = session_data.get(key)
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "(no fields collected yet)"
