"""Streamlit app for prompt-search-based ad generation."""

import base64
import json
import math
import mimetypes
import os
from typing import Any

import streamlit as st

from agent import (
    build_generation_prompt,
    chat,
    extract_fields,
    get_openai_client,
    try_parse_brief,
)
from generate import generate_image_gemini, generate_image_openai
from schema import STYLE_DIR, STYLE_PRESETS

TEXT_MODEL = "gpt-5-mini"
EVAL_MODEL = "gpt-4o"
DEFAULT_INITIAL_PROMPTS = 10
DEFAULT_LOWEST_PROMPTS = 5
DEFAULT_OPTIMIZATION_STEPS = 10
DEFAULT_EFFICIENT_CANDIDATES = 2
DEFAULT_JUDGE_REPEATS = 1


def _one_hot_probs(selected: str | None) -> dict[str, float]:
    """Create a one-hot probability distribution for 1-5."""
    if selected not in {"1", "2", "3", "4", "5"}:
        return {str(i): 0.2 for i in range(1, 6)}
    return {str(i): (1.0 if str(i) == selected else 0.0) for i in range(1, 6)}


def _parse_score_digit(text: str) -> str | None:
    """Extract a score digit from model text."""
    stripped = (text or "").strip()
    return stripped if stripped in {"1", "2", "3", "4", "5"} else None


def _parse_json_text(text: str) -> dict[str, Any]:
    """Parse JSON from a model response with simple fence stripping."""
    raw = (text or "").strip()
    if not raw:
        return {}

    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    elif "```" in raw:
        start = raw.index("```") + 3
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    elif "{" in raw and "}" in raw:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def _request_json_object(
    client,
    system_prompt: str,
    user_prompt: str | list[dict[str, Any]],
    *,
    temperature: float = 1.0,
    max_completion_tokens: int = 8192,
) -> dict[str, Any]:
    """Request a JSON object from GPT-5-mini."""
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    return _parse_json_text(response.choices[0].message.content or "{}")


def _prompt_excerpt(prompt: str, max_chars: int = 180) -> str:
    """Short prompt preview for UI and meta-reflection."""
    compact = " ".join((prompt or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for dedupe and display."""
    return " ".join((prompt or "").split())


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    """Deduplicate prompts while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for prompt in prompts:
        cleaned = _normalize_prompt(prompt)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _path_to_data_url(path: str) -> str:
    """Convert a local image path into a data URL for multimodal judging."""
    mime, _ = mimetypes.guess_type(path)
    subtype = "png" if mime is None else mime.split("/")[-1]
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{subtype};base64,{encoded}"


def _campaign_context_text(
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
) -> str:
    """Format the campaign context for generation and judging."""
    brief_json = json.dumps(creative_brief, indent=2, ensure_ascii=False)
    lines = [
        f"Product / Service: {session_data.get('product_name', '')}",
        f"Target Audience: {session_data.get('target_audience', '')}",
        f"Campaign Goal: {session_data.get('campaign_goal', '')}",
        f"Key Message / CTA: {session_data.get('key_message', '')}",
        f"Brand Tone: {session_data.get('brand_tone', '')}",
        f"Style Reference: {session_data.get('style_reference', '')}",
        f"Model: {sidebar_settings.get('model', '')}",
        f"Resolution: {sidebar_settings.get('resolution', '')}",
        f"Style Direction: {sidebar_settings.get('style_description', '')}",
        "",
        "Approved Creative Brief:",
        brief_json,
    ]
    return "\n".join(lines)


def _history_summary(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 3,
    bottom_n: int = 3,
) -> str:
    """Summarize the best and worst prompts in a history list."""
    if not candidates:
        return "No prior history."

    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0))
    bottom = sorted_candidates[:bottom_n]
    top = list(reversed(sorted_candidates[-top_n:]))

    lines = ["Highest-scoring prompts:"]
    if top:
        for entry in top:
            lines.append(
                f"- score={entry.get('score', 0.0):.3f} | {_prompt_excerpt(entry.get('prompt', ''))}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Lowest-scoring prompts:")
    if bottom:
        for entry in bottom:
            lines.append(
                f"- score={entry.get('score', 0.0):.3f} | {_prompt_excerpt(entry.get('prompt', ''))}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines)


def _reflection_examples(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 4,
    bottom_n: int = 4,
) -> list[dict[str, Any]]:
    """Select ranked high/low examples for multimodal reflection."""
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0))
    ranked_candidates = list(enumerate(sorted_candidates, start=1))
    total = len(ranked_candidates)
    examples: list[dict[str, Any]] = []

    if total < top_n + bottom_n:
        midpoint = total // 2
        lower_half = ranked_candidates[:midpoint]
        upper_half = ranked_candidates[midpoint:]

        for overall_rank, entry in lower_half:
            examples.append(
                {
                    "entry": entry,
                    "overall_rank": overall_rank,
                    "total": total,
                    "performance_label": "LOWER HALF",
                }
            )
        for overall_rank, entry in upper_half:
            examples.append(
                {
                    "entry": entry,
                    "overall_rank": overall_rank,
                    "total": total,
                    "performance_label": "UPPER HALF",
                }
            )
        return examples

    bottom_examples = ranked_candidates[:bottom_n]
    top_examples = list(reversed(ranked_candidates[-top_n:]))

    for cohort_rank, (overall_rank, entry) in enumerate(bottom_examples, start=1):
        examples.append(
            {
                "entry": entry,
                "overall_rank": overall_rank,
                "total": total,
                "performance_label": f"{cohort_rank} WORST PERFORMING",
            }
        )

    for cohort_rank, (overall_rank, entry) in enumerate(top_examples, start=1):
        examples.append(
            {
                "entry": entry,
                "overall_rank": overall_rank,
                "total": total,
                "performance_label": f"{cohort_rank} BEST PERFORMING",
            }
        )

    return examples


def _build_multimodal_reflection_content(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 4,
    bottom_n: int = 4,
) -> list[dict[str, Any]]:
    """Build multimodal reflection content from ranked prompt/image history."""
    examples = _reflection_examples(candidates, top_n=top_n, bottom_n=bottom_n)
    if not examples:
        return []

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are analyzing historical ad optimization results for one campaign.\n\n"
                "You will receive lower-performing and higher-performing examples. Each "
                "example includes an overall rank, score, prompt excerpt, and the rendered "
                "ad image when available.\n\n"
                "Infer visual and prompt-level patterns that tend to help or hurt performance. "
                "Focus on actionable differences between strong and weak ads.\n\n"
                'Return JSON with one key: "reflection". The reflection should be one short '
                "paragraph."
            ),
        }
    ]

    for example in examples:
        entry = example["entry"]
        output_path = str(entry.get("output_path") or "").strip()
        example_text = (
            f"{example['performance_label']} "
            f"(Rank #{example['overall_rank']}/{example['total']}, "
            f"Score: {entry.get('score', 0.0):.3f})\n"
            f"Prompt excerpt: {_prompt_excerpt(entry.get('prompt', ''), max_chars=240)}"
        )
        content.append({"type": "text", "text": example_text})

        if output_path and os.path.exists(output_path):
            try:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _path_to_data_url(output_path)},
                    }
                )
            except Exception:
                content.append(
                    {
                        "type": "text",
                        "text": "[Image could not be loaded for this example.]",
                    }
                )
        else:
            content.append(
                {
                    "type": "text",
                    "text": "[Rendered image not available for this example.]",
                }
            )

    return content


def _score_from_probs(probs: dict[str, float]) -> float:
    """Convert 1-5 probabilities into an expected score."""
    return sum(int(k) * probs[k] for k in sorted(probs.keys()))


def _extract_digit_probs_from_completion(response) -> dict[str, float]:
    """Extract 1-5 token probabilities from OpenAI chat logprobs."""
    probs = {str(i): 0.0 for i in range(1, 6)}
    choice = response.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None

    if content:
        token_info = content[0]
        for top in getattr(token_info, "top_logprobs", []) or []:
            token = (top.token or "").strip()
            if token in probs:
                probs[token] = math.exp(top.logprob)
        chosen_token = (getattr(token_info, "token", "") or "").strip()
        chosen_logprob = getattr(token_info, "logprob", None)
        if chosen_token in probs and probs[chosen_token] == 0.0 and chosen_logprob is not None:
            probs[chosen_token] = math.exp(chosen_logprob)

    total = sum(probs.values())
    if total <= 0:
        content_text = (choice.message.content or "").strip()
        if content_text in probs:
            probs[content_text] = 1.0
            total = 1.0

    if total <= 0:
        return {str(i): 0.2 for i in range(1, 6)}

    return {key: value / total for key, value in probs.items()}


def _score_prompt_with_logprobs(
    client,
    prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Text-only pre-score for prompt selection."""
    judge_prompt = f"""You are rating an image-generation prompt for a single advertising campaign.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Candidate prompt:
{prompt}

Return exactly one token from ["1","2","3","4","5"].

Scale:
1 = very unlikely to produce an effective ad for this campaign
2 = somewhat unlikely to produce an effective ad
3 = mixed / uncertain
4 = likely to produce an effective ad
5 = highly likely to produce an effective ad

Judge based on likely audience fit, clarity, visual specificity, brand-tone alignment, CTA support, and style adherence.
Do not explain your answer."""

    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=seed,
        )
        probs = _extract_digit_probs_from_completion(response)
    except Exception:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_completion_tokens=8,
            seed=seed,
        )
        probs = _one_hot_probs(_parse_score_digit(response.choices[0].message.content or ""))
    return {
        "score": _score_from_probs(probs),
        "probs": probs,
        "mode": "prompt",
    }


def _score_image_with_logprobs(
    client,
    image_path: str,
    prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Multimodal score for a rendered advertisement."""
    judge_text = f"""Evaluate this advertisement candidate for the following campaign.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Rendering prompt:
{prompt}

Return exactly one token from ["1","2","3","4","5"].

Scale:
1 = ineffective and poorly aligned with the campaign
2 = weak and unlikely to engage the target audience
3 = mixed / uncertain effectiveness
4 = strong and likely to engage the target audience
5 = excellent fit and highly likely to perform well

Judge based on overall ad effectiveness, audience fit, visual clarity, message delivery, CTA support, and style consistency.
Do not explain your answer."""

    image_url = _path_to_data_url(image_path)

    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=seed,
        )
        probs = _extract_digit_probs_from_completion(response)
        mode = "image"
    except Exception:
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": judge_text},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0,
                max_completion_tokens=8,
                seed=seed,
            )
            probs = _one_hot_probs(_parse_score_digit(response.choices[0].message.content or ""))
            mode = "image"
        except Exception:
            fallback = _score_prompt_with_logprobs(
                client,
                prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed,
            )
            probs = fallback["probs"]
            mode = "prompt-fallback"
    return {
        "score": _score_from_probs(probs),
        "probs": probs,
        "mode": mode,
    }


def _aggregate_soft_scores(
    scorer,
    *,
    repeats: int,
    seed_base: int | None = None,
) -> dict[str, Any]:
    """Average probability distributions from repeated scoring calls."""
    all_probs: list[dict[str, float]] = []
    mode = None
    for idx in range(max(1, repeats)):
        try:
            result = scorer(seed=None if seed_base is None else seed_base + idx)
        except Exception:
            result = {"probs": {str(i): 0.2 for i in range(1, 6)}, "mode": "fallback"}
        all_probs.append(result["probs"])
        mode = result.get("mode")

    averaged = {
        key: sum(prob[key] for prob in all_probs) / len(all_probs)
        for key in all_probs[0].keys()
    }
    return {
        "score": _score_from_probs(averaged),
        "probs": averaged,
        "mode": mode,
    }


def _generate_initial_prompt_variants(
    client,
    base_prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    total_count: int,
) -> list[str]:
    """Generate the starting set of prompt variants."""
    if total_count <= 1:
        return [base_prompt]

    campaign_context = _campaign_context_text(session_data, sidebar_settings, creative_brief)
    system_prompt = (
        "You generate polished image-generation prompts for advertising creatives. "
        "Return JSON only."
    )
    user_prompt = f"""Create {total_count - 1} additional image-generation prompts for the same campaign.

Locked campaign context:
{campaign_context}

Approved base prompt:
{base_prompt}

Requirements:
- Keep the same product, audience, campaign goal, core message, tone, style direction, and aspect ratio.
- Vary composition, scene setup, focal point, copy integration, camera framing, and visual emphasis enough to create genuinely different ad directions.
- Each prompt must be self-contained and ready for image generation.
- Keep each prompt concise and under 220 words.
- Return JSON with one key: "prompts".
"""

    payload = _request_json_object(
        client,
        system_prompt,
        user_prompt,
        temperature=1.0,
        max_completion_tokens=8192,
    )

    prompts = [base_prompt]
    if isinstance(payload.get("prompts"), list):
        prompts.extend([str(item) for item in payload["prompts"] if isinstance(item, str)])

    prompts = _dedupe_prompts(prompts)
    while len(prompts) < total_count:
        prompts.append(base_prompt)
    return prompts[:total_count]


def _generate_base_revision(
    client,
    current_prompt: str,
    current_score: float,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    chain_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate one prompt revision for the base optimizer."""
    history_text = _history_summary(chain_history, top_n=2, bottom_n=2)
    system_prompt = "You improve image-generation prompts for ad performance. Return JSON only."
    user_prompt = f"""Revise the current ad prompt for a better result.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Current score: {current_score:.3f}
Current prompt:
{current_prompt}

Recent local history:
{history_text}

Task:
- Produce exactly one revised prompt that preserves the campaign intent.
- Make a focused change rather than rewriting the campaign from scratch.
- Keep the prompt coherent, vivid, and under 220 words.
- Return JSON with keys "prompt" and "strategy".
"""

    payload = _request_json_object(
        client,
        system_prompt,
        user_prompt,
        temperature=1.0,
        max_completion_tokens=4096,
    )
    prompt = _normalize_prompt(str(payload.get("prompt", "")).strip()) or current_prompt
    strategy = str(payload.get("strategy", "")).strip()
    return {"prompt": prompt, "strategy": strategy}


def _generate_shared_reflection(
    client,
    shared_history: list[dict[str, Any]],
) -> str:
    """Summarize shared prompt patterns for the efficient optimizer."""
    if len(shared_history) < 2:
        return "No strong shared pattern yet."

    def _text_only_reflection() -> str:
        system_prompt = "You analyze prompt-performance patterns. Return JSON only."
        user_prompt = f"""Review the prompt history below and infer what tends to help or hurt performance.

{_history_summary(shared_history, top_n=4, bottom_n=4)}

Return JSON with one key: "reflection".
The reflection should be one short paragraph focused on actionable prompt-level patterns.
"""

        payload = _request_json_object(
            client,
            system_prompt,
            user_prompt,
            temperature=0.7,
            max_completion_tokens=2048,
        )
        reflection = str(payload.get("reflection", "")).strip()
        return reflection or "No strong shared pattern yet."

    multimodal_content = _build_multimodal_reflection_content(
        shared_history,
        top_n=4,
        bottom_n=4,
    )
    if not multimodal_content:
        return _text_only_reflection()

    system_prompt = "You analyze ad-performance patterns from prompt/image history. Return JSON only."
    try:
        payload = _request_json_object(
            client,
            system_prompt,
            multimodal_content,
            temperature=0.7,
            max_completion_tokens=2048,
        )
    except Exception:
        return _text_only_reflection()

    reflection = str(payload.get("reflection", "")).strip()
    return reflection or _text_only_reflection()


def _generate_efficient_candidates(
    client,
    current_prompt: str,
    current_score: float,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    shared_history: list[dict[str, Any]],
    shared_reflection: str,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    """Generate multiple prompt candidates for the efficient optimizer."""
    system_prompt = (
        "You propose multiple high-upside prompt revisions for ad creative generation. "
        "Return JSON only."
    )
    user_prompt = f"""Create {candidate_count} revised prompt candidates for the current trajectory.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Current score: {current_score:.3f}
Current prompt:
{current_prompt}

Shared reflection:
{shared_reflection}

Shared history summary:
{_history_summary(shared_history, top_n=3, bottom_n=3)}

Requirements:
- Each candidate must stay faithful to the same campaign.
- Candidates should be meaningfully distinct from one another.
- Use the shared reflection to avoid weak patterns and explore stronger ones.
- Keep each prompt coherent and under 220 words.
- Return JSON with keys "strategy" and "prompts".
"""

    payload = _request_json_object(
        client,
        system_prompt,
        user_prompt,
        temperature=1.0,
        max_completion_tokens=4096,
    )

    prompts: list[str] = []
    if isinstance(payload.get("prompts"), list):
        prompts = [str(item) for item in payload["prompts"] if isinstance(item, str)]
    prompts = _dedupe_prompts(prompts)
    if not prompts:
        prompts = [current_prompt]

    while len(prompts) < candidate_count:
        prompts.append(prompts[-1])

    return {
        "strategy": str(payload.get("strategy", "")).strip(),
        "prompts": prompts[:candidate_count],
    }


def _generate_image_for_prompt(
    *,
    client,
    gemini_key: str,
    model_key: str,
    prompt: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
) -> tuple[str | None, str | None]:
    """Generate an image using the selected backend."""
    if model_key == "gemini":
        output_path, error = generate_image_gemini(
            gemini_key,
            prompt,
            style_image_bytes=style_image_bytes,
        )
    else:
        size_map = {
            "1536x1024 (landscape)": "1536x1024",
            "1024x1024 (square)": "1024x1024",
            "1024x1536 (portrait)": "1024x1536",
        }
        output_path, error = generate_image_openai(
            client,
            prompt,
            size=size_map.get(aspect_ratio, "1536x1024"),
            quality=gpt_image_quality,
        )

    return (str(output_path), error) if output_path else (None, error)


def _build_candidate_record(
    *,
    candidate_id: str,
    prompt: str,
    output_path: str | None,
    score: float,
    probs: dict[str, float],
    source: str,
    error: str | None = None,
    prompt_prescore: float | None = None,
    prompt_prescore_probs: dict[str, float] | None = None,
    strategy: str | None = None,
    accepted: bool | None = None,
    start_seed_id: str | None = None,
    start_seed_rank: int | None = None,
    step: int | None = None,
    chain_id: int | None = None,
    trajectory_id: int | None = None,
) -> dict[str, Any]:
    """Create a consistent candidate record."""
    return {
        "candidate_id": candidate_id,
        "prompt": prompt,
        "output_path": output_path,
        "score": score,
        "probs": probs,
        "source": source,
        "error": error,
        "prompt_prescore": prompt_prescore,
        "prompt_prescore_probs": prompt_prescore_probs,
        "strategy": strategy,
        "accepted": accepted,
        "start_seed_id": start_seed_id,
        "start_seed_rank": start_seed_rank,
        "step": step,
        "chain_id": chain_id,
        "trajectory_id": trajectory_id,
    }


def _rank_candidates_desc(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidates by score descending."""
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)


def _run_base_optimizer(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    worst_seeds: list[dict[str, Any]],
    optimization_steps: int,
    judge_repeats: int,
    report,
) -> dict[str, Any]:
    """Run the base optimization chains independently from the worst seeds."""
    chains: list[dict[str, Any]] = []
    all_evaluated: list[dict[str, Any]] = []

    for chain_idx, seed in enumerate(worst_seeds, start=1):
        report(
            f"Base chain {chain_idx}/{len(worst_seeds)}: starting from score {seed['score']:.3f}"
        )
        current = dict(seed)
        best = dict(seed)
        chain_history = [dict(seed)]
        step_entries: list[dict[str, Any]] = []

        for step in range(1, optimization_steps + 1):
            revision = _generate_base_revision(
                openai_client,
                current["prompt"],
                current["score"],
                session_data,
                sidebar_settings,
                creative_brief,
                chain_history,
            )
            new_prompt = revision["prompt"]
            output_path, error = _generate_image_for_prompt(
                client=openai_client,
                gemini_key=gemini_key,
                model_key=model_key,
                prompt=new_prompt,
                aspect_ratio=aspect_ratio,
                gpt_image_quality=gpt_image_quality,
                style_image_bytes=style_image_bytes,
            )

            if output_path:
                scored = _aggregate_soft_scores(
                    lambda seed_value: _score_image_with_logprobs(
                        openai_client,
                        output_path,
                        new_prompt,
                        session_data,
                        sidebar_settings,
                        creative_brief,
                        seed=seed_value,
                    ),
                    repeats=judge_repeats,
                    seed_base=chain_idx * 1000 + step * 10,
                )
                score = scored["score"]
                probs = scored["probs"]
            else:
                score = 1.0
                probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}

            accepted = score > current["score"]
            step_entry = _build_candidate_record(
                candidate_id=f"base_chain{chain_idx:02d}_step{step:02d}",
                prompt=new_prompt,
                output_path=output_path,
                score=score,
                probs=probs,
                source="base",
                error=error,
                strategy=revision.get("strategy"),
                accepted=accepted,
                start_seed_id=seed["candidate_id"],
                start_seed_rank=seed.get("worst_rank"),
                step=step,
                chain_id=chain_idx,
            )
            step_entries.append(step_entry)
            chain_history.append(step_entry)
            all_evaluated.append(step_entry)

            if accepted:
                current = dict(step_entry)
            if score > best["score"]:
                best = dict(step_entry)

        chains.append(
            {
                "chain_id": chain_idx,
                "seed": seed,
                "steps": step_entries,
                "final": current,
                "best": best,
            }
        )

    best_overall = max([chain["best"] for chain in chains], key=lambda item: item["score"])
    return {
        "chains": chains,
        "all_evaluated": all_evaluated,
        "best": best_overall,
    }


def _run_efficient_optimizer(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    worst_seeds: list[dict[str, Any]],
    optimization_steps: int,
    efficient_candidates: int,
    judge_repeats: int,
    report,
) -> dict[str, Any]:
    """Run the shared-history efficient optimizer from the same worst seeds."""
    trajectories: list[dict[str, Any]] = []
    shared_history: list[dict[str, Any]] = [dict(seed) for seed in worst_seeds]
    shared_reflection = _generate_shared_reflection(openai_client, shared_history)
    all_evaluated: list[dict[str, Any]] = []
    shared_reflection_history = [
        {"step": 0, "reflection": shared_reflection, "history_size": len(shared_history)}
    ]

    for trajectory_idx, seed in enumerate(worst_seeds, start=1):
        trajectories.append(
            {
                "trajectory_id": trajectory_idx,
                "seed": seed,
                "current": dict(seed),
                "best": dict(seed),
                "steps": [],
            }
        )

    for step in range(1, optimization_steps + 1):
        report(
            f"Efficient step {step}/{optimization_steps}: shared reflection refreshed across "
            f"{len(shared_history)} evaluated prompts"
        )

        for trajectory in trajectories:
            current = trajectory["current"]
            proposals = _generate_efficient_candidates(
                openai_client,
                current["prompt"],
                current["score"],
                session_data,
                sidebar_settings,
                creative_brief,
                shared_history,
                shared_reflection,
                candidate_count=efficient_candidates,
            )

            prescored_candidates: list[dict[str, Any]] = []
            for cand_idx, candidate_prompt in enumerate(proposals["prompts"], start=1):
                prompt_scored = _aggregate_soft_scores(
                    lambda seed_value: _score_prompt_with_logprobs(
                        openai_client,
                        candidate_prompt,
                        session_data,
                        sidebar_settings,
                        creative_brief,
                        seed=seed_value,
                    ),
                    repeats=judge_repeats,
                    seed_base=step * 10000 + trajectory["trajectory_id"] * 100 + cand_idx,
                )
                prescored_candidates.append(
                    {
                        "prompt": candidate_prompt,
                        "score": prompt_scored["score"],
                        "probs": prompt_scored["probs"],
                    }
                )

            selected_prompt_entry = max(prescored_candidates, key=lambda item: item["score"])
            selected_prompt = selected_prompt_entry["prompt"]

            output_path, error = _generate_image_for_prompt(
                client=openai_client,
                gemini_key=gemini_key,
                model_key=model_key,
                prompt=selected_prompt,
                aspect_ratio=aspect_ratio,
                gpt_image_quality=gpt_image_quality,
                style_image_bytes=style_image_bytes,
            )

            if output_path:
                image_scored = _aggregate_soft_scores(
                    lambda seed_value: _score_image_with_logprobs(
                        openai_client,
                        output_path,
                        selected_prompt,
                        session_data,
                        sidebar_settings,
                        creative_brief,
                        seed=seed_value,
                    ),
                    repeats=judge_repeats,
                    seed_base=step * 20000 + trajectory["trajectory_id"] * 100,
                )
                final_score = image_scored["score"]
                final_probs = image_scored["probs"]
            else:
                final_score = 1.0
                final_probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}

            accepted = final_score > current["score"]
            step_entry = _build_candidate_record(
                candidate_id=(
                    f"efficient_traj{trajectory['trajectory_id']:02d}_step{step:02d}"
                ),
                prompt=selected_prompt,
                output_path=output_path,
                score=final_score,
                probs=final_probs,
                source="efficient",
                error=error,
                prompt_prescore=selected_prompt_entry["score"],
                prompt_prescore_probs=selected_prompt_entry["probs"],
                strategy=proposals.get("strategy"),
                accepted=accepted,
                start_seed_id=trajectory["seed"]["candidate_id"],
                start_seed_rank=trajectory["seed"].get("worst_rank"),
                step=step,
                trajectory_id=trajectory["trajectory_id"],
            )
            step_entry["candidate_prompts"] = prescored_candidates
            step_entry["shared_reflection"] = shared_reflection
            trajectory["steps"].append(step_entry)
            shared_history.append(step_entry)
            all_evaluated.append(step_entry)

            if accepted:
                trajectory["current"] = dict(step_entry)
            if final_score > trajectory["best"]["score"]:
                trajectory["best"] = dict(step_entry)

        shared_reflection = _generate_shared_reflection(openai_client, shared_history)
        shared_reflection_history.append(
            {
                "step": step,
                "reflection": shared_reflection,
                "history_size": len(shared_history),
            }
        )

    best_overall = max([trajectory["best"] for trajectory in trajectories], key=lambda item: item["score"])
    return {
        "trajectories": trajectories,
        "all_evaluated": all_evaluated,
        "best": best_overall,
        "shared_reflection_history": shared_reflection_history,
    }


def _run_search_pipeline(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    initial_prompt_count: int,
    lowest_prompt_count: int,
    optimization_steps: int,
    efficient_candidates: int,
    judge_repeats: int,
    report,
) -> dict[str, Any]:
    """Run the full prompt-search pipeline."""
    base_prompt = build_generation_prompt(
        creative_brief,
        session_data,
        sidebar_settings=sidebar_settings,
        style_description=sidebar_settings.get("style_description", ""),
        has_style_image=(model_key == "gemini" and style_image_bytes is not None),
    )
    report(f"Base prompt ready ({len(base_prompt.split())} words)")

    prompt_variants = _generate_initial_prompt_variants(
        openai_client,
        base_prompt,
        session_data,
        sidebar_settings,
        creative_brief,
        total_count=initial_prompt_count,
    )
    report(f"Prepared {len(prompt_variants)} initial prompt candidates")

    initial_candidates: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompt_variants, start=1):
        report(f"Initial candidate {idx}/{len(prompt_variants)}: generating image")
        output_path, error = _generate_image_for_prompt(
            client=openai_client,
            gemini_key=gemini_key,
            model_key=model_key,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            gpt_image_quality=gpt_image_quality,
            style_image_bytes=style_image_bytes,
        )

        if output_path:
            scored = _aggregate_soft_scores(
                lambda seed_value: _score_image_with_logprobs(
                    openai_client,
                    output_path,
                    prompt,
                    session_data,
                    sidebar_settings,
                    creative_brief,
                    seed=seed_value,
                ),
                repeats=judge_repeats,
                seed_base=idx * 100,
            )
            score = scored["score"]
            probs = scored["probs"]
        else:
            score = 1.0
            probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}

        initial_candidates.append(
            _build_candidate_record(
                candidate_id=f"initial_{idx:02d}",
                prompt=prompt,
                output_path=output_path,
                score=score,
                probs=probs,
                source="initial",
                error=error,
            )
        )

    ranked_initial = _rank_candidates_desc(initial_candidates)
    successful_initial = [item for item in initial_candidates if item.get("output_path")]
    if not successful_initial:
        raise RuntimeError("All initial image generations failed.")

    worst_pool = sorted(successful_initial, key=lambda item: item["score"])[:lowest_prompt_count]
    for rank, entry in enumerate(worst_pool, start=1):
        entry["worst_rank"] = rank

    report(
        f"Selected {len(worst_pool)} lowest-scoring starting points for both optimizers"
    )

    base_results = _run_base_optimizer(
        openai_client=openai_client,
        gemini_key=gemini_key,
        model_key=model_key,
        aspect_ratio=aspect_ratio,
        gpt_image_quality=gpt_image_quality,
        style_image_bytes=style_image_bytes,
        session_data=session_data,
        sidebar_settings=sidebar_settings,
        creative_brief=creative_brief,
        worst_seeds=worst_pool,
        optimization_steps=optimization_steps,
        judge_repeats=judge_repeats,
        report=report,
    )

    efficient_results = _run_efficient_optimizer(
        openai_client=openai_client,
        gemini_key=gemini_key,
        model_key=model_key,
        aspect_ratio=aspect_ratio,
        gpt_image_quality=gpt_image_quality,
        style_image_bytes=style_image_bytes,
        session_data=session_data,
        sidebar_settings=sidebar_settings,
        creative_brief=creative_brief,
        worst_seeds=worst_pool,
        optimization_steps=optimization_steps,
        efficient_candidates=efficient_candidates,
        judge_repeats=judge_repeats,
        report=report,
    )

    overall_best = max(
        [
            max(ranked_initial, key=lambda item: item["score"]),
            base_results["best"],
            efficient_results["best"],
        ],
        key=lambda item: item["score"],
    )

    return {
        "base_prompt": base_prompt,
        "creative_brief": creative_brief,
        "initial_candidates": initial_candidates,
        "initial_ranked": ranked_initial,
        "worst_pool": worst_pool,
        "base": base_results,
        "efficient": efficient_results,
        "overall_best": overall_best,
        "config": {
            "initial_prompt_count": initial_prompt_count,
            "lowest_prompt_count": lowest_prompt_count,
            "optimization_steps": optimization_steps,
            "efficient_candidates": efficient_candidates,
            "judge_repeats": judge_repeats,
            "model": sidebar_settings.get("model"),
            "resolution": sidebar_settings.get("resolution"),
        },
    }


def _render_probs(probs: dict[str, float]) -> str:
    """Render a compact probability summary."""
    return " | ".join(f"{k}: {probs.get(k, 0.0):.2f}" for k in ["1", "2", "3", "4", "5"])


def _render_candidate_gallery(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    key_prefix: str,
    sort_desc: bool = True,
) -> None:
    """Render a gallery of candidates with images, prompts, and scores."""
    st.subheader(title)
    items = _rank_candidates_desc(candidates) if sort_desc else list(candidates)
    if not items:
        st.info("No candidates available.")
        return

    for idx in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for col, candidate in zip(cols, items[idx : idx + 2]):
            with col:
                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown(
                    f"**Score:** {candidate['score']:.3f}  \n"
                    f"**Source:** {candidate['source']}"
                )
                if candidate.get("prompt_prescore") is not None:
                    st.caption(f"Prompt pre-score: {candidate['prompt_prescore']:.3f}")
                if candidate.get("output_path"):
                    st.image(candidate["output_path"], use_container_width=True)
                else:
                    st.warning(candidate.get("error") or "Image generation failed.")
                with st.expander("Prompt", expanded=False):
                    st.code(candidate["prompt"], language="text")
                st.caption(_render_probs(candidate["probs"]))
                if candidate.get("strategy"):
                    st.markdown(f"**Strategy:** {candidate['strategy']}")
                st.markdown("</div>", unsafe_allow_html=True)


def _render_base_results(results: dict[str, Any]) -> None:
    """Render base optimizer results."""
    st.subheader("Base Search")
    best = results["best"]
    st.markdown(
        f"Best base result: **{best['score']:.3f}** from seed `{best.get('start_seed_id', 'n/a')}`."
    )

    for chain in results["chains"]:
        seed = chain["seed"]
        best_chain = chain["best"]
        with st.expander(
            f"Chain {chain['chain_id']} | seed score {seed['score']:.3f} | "
            f"best {best_chain['score']:.3f}",
            expanded=False,
        ):
            st.markdown(f"**Seed prompt:** `{seed['candidate_id']}`")
            if seed.get("output_path"):
                st.image(seed["output_path"], use_container_width=True)
            st.code(seed["prompt"], language="text")
            if chain["steps"]:
                _render_candidate_gallery(
                    "Step Results",
                    chain["steps"],
                    key_prefix=f"base_chain_{chain['chain_id']}",
                    sort_desc=False,
                )


def _render_efficient_results(results: dict[str, Any]) -> None:
    """Render efficient optimizer results."""
    st.subheader("Efficient Search")
    best = results["best"]
    st.markdown(
        f"Best efficient result: **{best['score']:.3f}** from seed `{best.get('start_seed_id', 'n/a')}`."
    )

    with st.expander("Shared Reflections", expanded=False):
        for item in results["shared_reflection_history"]:
            st.markdown(
                f"**Step {item['step']}**  \n"
                f"History size: {item['history_size']}  \n"
                f"{item['reflection']}"
            )

    for trajectory in results["trajectories"]:
        seed = trajectory["seed"]
        best_traj = trajectory["best"]
        with st.expander(
            f"Trajectory {trajectory['trajectory_id']} | seed score {seed['score']:.3f} | "
            f"best {best_traj['score']:.3f}",
            expanded=False,
        ):
            st.markdown(f"**Seed prompt:** `{seed['candidate_id']}`")
            if seed.get("output_path"):
                st.image(seed["output_path"], use_container_width=True)
            st.code(seed["prompt"], language="text")

            for step_entry in trajectory["steps"]:
                st.markdown(
                    f"**Step {step_entry['step']}** | "
                    f"pre-score {step_entry.get('prompt_prescore', 0.0):.3f} | "
                    f"final score {step_entry['score']:.3f}"
                )
                if step_entry.get("output_path"):
                    st.image(step_entry["output_path"], use_container_width=True)
                if step_entry.get("candidate_prompts"):
                    with st.expander(
                        f"Prompt Candidates For Step {step_entry['step']}",
                        expanded=False,
                    ):
                        for cand in step_entry["candidate_prompts"]:
                            st.markdown(
                                f"- pre-score {cand['score']:.3f} | "
                                f"{_prompt_excerpt(cand['prompt'], 260)}"
                            )
                st.code(step_entry["prompt"], language="text")
                if step_entry.get("strategy"):
                    st.caption(step_entry["strategy"])


# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Campaign Agent Optimizer",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .stChatMessage {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .style-selected-ring {
        border: 2.5px solid #6366f1;
        border-radius: 10px;
        padding: 2px;
    }
    .style-default-ring {
        border: 2.5px solid transparent;
        border-radius: 10px;
        padding: 2px;
    }
    .phase-step {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.3rem 0; font-size: 0.85rem; color: #94a3b8;
    }
    .phase-step.active { color: #6366f1; font-weight: 600; }
    .phase-step.done   { color: #22c55e; }
    .phase-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #cbd5e1; flex-shrink: 0;
    }
    .phase-step.active .phase-dot { background: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.25); }
    .phase-step.done .phase-dot { background: #22c55e; }
    .output-card {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin: 0.75rem 0;
    }
    .welcome-hero {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 50%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .welcome-hero h2 { color: white; margin: 0 0 0.5rem 0; font-size: 1.5rem; }
    .welcome-hero p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 0.95rem; line-height: 1.5; }
    .approve-bar {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .approve-bar p {
        margin: 0;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.4;
    }
    .model-note {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Ad Campaign Agent Optimizer")
st.caption("GPT-5-mini + prompt search + image generation")

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("API Keys")

    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    gemini_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        key="gemini_key_input",
    )

    if not openai_key:
        st.warning("OpenAI API key required.")
        st.stop()

    st.divider()
    st.header("Generation Settings")

    image_models = {"GPT Image 1.5": "gpt"}
    if gemini_key:
        image_models["Gemini 2.5 Flash"] = "gemini"

    model_label = st.selectbox("Model", list(image_models.keys()))
    model_key = image_models[model_label]

    gpt_quality = "medium"
    if model_key == "gemini":
        st.markdown(
            '<div class="model-note">Multimodal LLM with native image output. '
            'Uses the style reference image directly for visual matching.</div>',
            unsafe_allow_html=True,
        )
        aspect_ratio = "auto"
    else:
        st.markdown(
            '<div class="model-note">OpenAI dedicated image generation API. '
            'Style is applied via text prompt only.</div>',
            unsafe_allow_html=True,
        )
        gpt_res = st.selectbox(
            "Resolution",
            ["1536x1024 (landscape)", "1024x1024 (square)", "1024x1536 (portrait)"],
        )
        aspect_ratio = gpt_res
        gpt_quality = st.selectbox(
            "Quality",
            ["low", "medium", "high"],
            index=1,
            format_func=lambda q: q.capitalize(),
        )

    gpt_image_quality = gpt_quality

    st.divider()
    st.header("Style Reference")

    style_keys_list = list(STYLE_PRESETS.keys())
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = style_keys_list[0]

    cols_per_row = 2
    for row_start in range(0, len(style_keys_list), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for col_idx, col in enumerate(cols):
            key_idx = row_start + col_idx
            if key_idx >= len(style_keys_list):
                break
            skey = style_keys_list[key_idx]
            preset = STYLE_PRESETS[skey]
            style_path = STYLE_DIR / f"{skey}.png"
            is_selected = st.session_state.selected_style == skey
            with col:
                ring_class = "style-selected-ring" if is_selected else "style-default-ring"
                if style_path.exists():
                    st.markdown(f'<div class="{ring_class}">', unsafe_allow_html=True)
                    st.image(str(style_path), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    preset["label"],
                    key=f"style_btn_{skey}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.selected_style = skey
                    st.rerun()

    selected_style_key = st.session_state.selected_style
    style_path = STYLE_DIR / f"{selected_style_key}.png"
    style_image_bytes = style_path.read_bytes() if style_path.exists() else None
    style_description = STYLE_PRESETS[selected_style_key]["description"]

    st.divider()
    st.header("Search Settings")
    initial_prompt_count = st.number_input(
        "Initial prompts",
        min_value=2,
        max_value=20,
        value=DEFAULT_INITIAL_PROMPTS,
        step=1,
    )
    lowest_prompt_count = st.number_input(
        "Lowest prompts to optimize",
        min_value=1,
        max_value=10,
        value=DEFAULT_LOWEST_PROMPTS,
        step=1,
    )
    optimization_steps = st.number_input(
        "Optimization steps",
        min_value=1,
        max_value=20,
        value=DEFAULT_OPTIMIZATION_STEPS,
        step=1,
    )
    efficient_candidates = st.number_input(
        "Efficient prompt candidates / step",
        min_value=1,
        max_value=5,
        value=DEFAULT_EFFICIENT_CANDIDATES,
        step=1,
    )

    sidebar_settings = {
        "model": model_label,
        "resolution": aspect_ratio if aspect_ratio != "auto" else "auto",
        "style_description": style_description,
    }

# ─── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "product_name": "",
        "target_audience": "",
        "campaign_goal": "",
        "key_message": "",
        "brand_tone": "",
        "style_reference": "",
    }

if "phase" not in st.session_state:
    st.session_state.phase = "collecting"

if "creative_brief" not in st.session_state:
    st.session_state.creative_brief = None

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

if style_image_bytes:
    st.session_state.session_data["style_reference"] = STYLE_PRESETS[selected_style_key]["label"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

openai_client = get_openai_client(openai_key)

if not st.session_state.messages:
    st.markdown(
        """
    <div class="welcome-hero">
        <h2>Welcome to Ad Campaign Agent Optimizer</h2>
        <p>Choose a model and visual style in the sidebar, then describe your campaign.
        After you approve the brief, the app will seed multiple prompts, score them,
        and run both base and efficient prompt-search loops.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    greeting = (
        "Hi! I'm your **Ad Campaign Agent Optimizer**. I'll help you build and search over ad prompts.\n\n"
        "To get started, tell me about your campaign. I'll need these details:\n\n"
        "- **Product / Service Name** — what you're advertising\n"
        "- **Target Audience** — who you're reaching\n"
        "- **Campaign Goal** — awareness, consideration, conversion, or launch\n"
        "- **Key Message / CTA** — the main takeaway and call to action\n"
        "- **Brand Tone** — the emotional feel (e.g. bold, calm, playful)\n\n"
        "You can share everything at once or one at a time — I'll guide you through it."
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.markdown(greeting)

trigger_generation = st.session_state.pop("trigger_generation", False)

if (
    st.session_state.phase == "reviewing"
    and st.session_state.creative_brief
    and not trigger_generation
):
    st.markdown(
        '<div class="approve-bar">'
        '<p>Brief is ready for review. Type feedback below to request changes, '
        'or click the button to run the prompt search.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        if st.button("Approve & Run", type="primary", use_container_width=True):
            st.session_state.trigger_generation = True
            st.rerun()


def _run_optimization() -> None:
    """Execute the prompt-search pipeline and persist results."""
    st.session_state.phase = "generating"

    try:
        with st.chat_message("assistant"):
            with st.status("Running prompt search...", expanded=True) as status:
                results = _run_search_pipeline(
                    openai_client=openai_client,
                    gemini_key=gemini_key,
                    model_key=model_key,
                    aspect_ratio=aspect_ratio,
                    gpt_image_quality=gpt_image_quality,
                    style_image_bytes=style_image_bytes,
                    session_data=st.session_state.session_data,
                    sidebar_settings=sidebar_settings,
                    creative_brief=st.session_state.creative_brief,
                    initial_prompt_count=int(initial_prompt_count),
                    lowest_prompt_count=int(lowest_prompt_count),
                    optimization_steps=int(optimization_steps),
                    efficient_candidates=int(efficient_candidates),
                    judge_repeats=DEFAULT_JUDGE_REPEATS,
                    report=st.write,
                )
                status.update(label="Prompt search complete!", state="complete", expanded=False)

        st.session_state.optimization_results = results
        st.session_state.phase = "done"
        response_text = (
            "Search finished. The initial prompts, the lowest-performing seeds, and the "
            "base and efficient optimizer results are shown below."
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
    except Exception as exc:
        st.session_state.phase = "reviewing"
        error_text = f"Search failed: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        st.error(error_text)


if trigger_generation and st.session_state.creative_brief:
    approval_msg = "Approved — run the prompt search."
    st.session_state.messages.append({"role": "user", "content": approval_msg})
    with st.chat_message("user"):
        st.markdown(approval_msg)
    _run_optimization()

phase = st.session_state.get("phase", "collecting")
placeholders = {
    "collecting": "Describe your product and campaign...",
    "reviewing": "Type feedback to revise the brief...",
    "done": "Start a new campaign (reset in sidebar)",
}
user_input = st.chat_input(placeholders.get(phase, "Describe your campaign idea..."))

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.phase == "done":
        done_text = "Reset the session from the sidebar to start a new campaign."
        st.session_state.messages.append({"role": "assistant", "content": done_text})
        with st.chat_message("assistant"):
            st.markdown(done_text)
        st.stop()

    approve_words = [
        "approve",
        "approved",
        "looks good",
        "let's go",
        "run",
        "yes",
        "proceed",
        "go ahead",
        "perfect",
        "love it",
        "lgtm",
        "do it",
        "go for it",
        "ship it",
        "make it",
        "create it",
        "好",
        "好的",
        "可以",
        "没问题",
        "开始",
        "生成",
        "确认",
        "批准",
        "通过",
        "同意",
        "行",
        "就这样",
        "没意见",
        "ok",
    ]
    is_approval = (
        st.session_state.phase == "reviewing"
        and any(word in user_input.lower().strip() for word in approve_words)
    )

    if is_approval and st.session_state.creative_brief:
        _run_optimization()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = chat(
                    openai_client,
                    st.session_state.messages,
                    st.session_state.session_data,
                    sidebar_settings=sidebar_settings,
                )

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            brief = try_parse_brief(response_text)
            if brief and (
                "composition" in brief or "headline" in brief or "visual_style" in brief
            ):
                st.session_state.creative_brief = brief
                st.session_state.phase = "reviewing"
                st.rerun()

            if st.session_state.phase == "collecting":
                try:
                    extracted = extract_fields(openai_client, st.session_state.messages)
                    for key, value in extracted.items():
                        if value and key in st.session_state.session_data:
                            st.session_state.session_data[key] = value
                except Exception:
                    pass

if st.session_state.optimization_results:
    results = st.session_state.optimization_results
    overall_best = results["overall_best"]

    st.divider()
    st.header("Search Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        best_initial = max(results["initial_candidates"], key=lambda item: item["score"])
        st.metric("Best Initial", f"{best_initial['score']:.3f}")
    with col2:
        st.metric("Best Base", f"{results['base']['best']['score']:.3f}")
    with col3:
        st.metric("Best Efficient", f"{results['efficient']['best']['score']:.3f}")

    st.markdown(
        f"**Overall best score:** {overall_best['score']:.3f} from `{overall_best['source']}`."
    )
    if overall_best.get("output_path"):
        st.image(overall_best["output_path"], use_container_width=True)
    with st.expander("Overall Best Prompt", expanded=False):
        st.code(overall_best["prompt"], language="text")

    download_payload = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Results JSON",
        data=download_payload,
        file_name="ad_campaign_optimizer_results.json",
        mime="application/json",
        use_container_width=False,
    )

    tabs = st.tabs(
        [
            "Initial",
            f"Lowest {len(results['worst_pool'])}",
            "Base",
            "Efficient",
        ]
    )
    with tabs[0]:
        _render_candidate_gallery(
            "Initial Prompt Candidates",
            results["initial_candidates"],
            key_prefix="initial_candidates",
        )
    with tabs[1]:
        _render_candidate_gallery(
            "Lowest-Scoring Starting Points",
            results["worst_pool"],
            key_prefix="worst_pool",
            sort_desc=False,
        )
    with tabs[2]:
        _render_base_results(results["base"])
    with tabs[3]:
        _render_efficient_results(results["efficient"])

with st.sidebar:
    st.divider()
    st.header("Session Status")

    current_phase = st.session_state.phase
    steps = [
        ("collecting", "Collect info"),
        ("reviewing", "Review brief"),
        ("generating", "Run search"),
        ("done", "Done"),
    ]
    phase_order = [item[0] for item in steps]
    current_idx = phase_order.index(current_phase) if current_phase in phase_order else 0

    stepper_html = ""
    for i, (_, phase_label) in enumerate(steps):
        if i < current_idx:
            cls = "phase-step done"
        elif i == current_idx:
            cls = "phase-step active"
        else:
            cls = "phase-step"
        stepper_html += f'<div class="{cls}"><span class="phase-dot"></span>{phase_label}</div>'

    st.markdown(stepper_html, unsafe_allow_html=True)

    if st.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
