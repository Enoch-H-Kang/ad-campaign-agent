# Ad Campaign Agent

An AI-powered ad campaign creative tool built with Streamlit. Describe your product and campaign idea — the agent collects requirements, generates a creative brief, and produces an ad image.

**Live demo:** [ad-campaign-agent.streamlit.app](https://ad-campaign-agent.streamlit.app)

## How It Works

1. **Choose a model** — GPT Image 1.5 (default) or Gemini 2.5 Flash
2. **Pick a visual style** — 6 curated style presets
3. **Describe your campaign** — product, audience, goal, message, tone
4. **Review the creative brief** — edit or approve
5. **Generate** — the agent builds an optimized prompt and generates your ad image

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture diagram" width="800">
</p>

### File Structure

```
app.py          Streamlit UI — chat interface, sidebar controls, generation flow
agent.py        GPT-5-mini agent — conversation, field extraction, brief parsing
prompts.py      System prompt — pipeline rules, brief format, behavior
generate.py     Image generation backends — OpenAI and Gemini APIs
schema.py       Field definitions, style presets, validation
style/          Style reference images (6 presets)
```

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- (Optional) A Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Install & Run

**Option A — Download ZIP (no Git needed, no account needed)**

1. Click the green **Code** button on this page > **Download ZIP**
2. Unzip the folder, open a terminal inside it, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Option B — Git clone (also no account needed)**

```bash
git clone https://github.com/zikunye2/ad-campaign-agent.git
cd ad-campaign-agent
pip install -r requirements.txt
streamlit run app.py
```

**Option C — Use a coding agent**

Give this repo URL to a coding agent like [Claude Code](https://claude.ai/claude-code), [Codex](https://chatgpt.com/codex), or [Cursor](https://cursor.com) — it will clone the repo, install dependencies, and run the app for you. From there you can ask the agent to add features, fix bugs, or modify the code.

```
https://github.com/zikunye2/ad-campaign-agent.git
```

All options work on **both macOS and Windows**. If `pip` doesn't work, try `python -m pip install -r requirements.txt`.

Enter your API key(s) in the sidebar and start chatting.

### API Key Safety

API keys are entered in the browser and stored **only in your session memory**. They are never saved to disk, logged, or sent anywhere except to the OpenAI/Google APIs. When you close the tab, keys are gone.

## Models

| Model | API | Style Reference | Notes |
|-------|-----|-----------------|-------|
| **GPT Image 1.5** | OpenAI | Text description only | 3 quality tiers (low/medium/high), configurable resolution |
| **Gemini 2.5 Flash** | Google | Image + text (native) | Multimodal LLM, uses style image directly |

## For Students

This repo is designed for you to download, run locally, and extend.

### Getting Started

1. **Download the code** — see [Install & Run](#install--run) above (ZIP or git clone, no account needed)
2. **Run it** — `pip install -r requirements.txt` then `streamlit run app.py`
3. **Make changes** — edit the Python files, save, and Streamlit auto-reloads
4. **Show your work** — screenshot the app, explain what you changed and why

> **New to Git?** Check out GitHub's official guide: [docs.github.com/en/get-started](https://docs.github.com/en/get-started/start-your-journey)

### Project Ideas

Some broad directions — be creative:

- **Better UI / UX** — redesign the interface, improve the workflow
- **Better prompts** — improve the system prompt, brief format, or generation prompt
- **New tools** — add video generation, image editing, multi-image A/B testing
- **Post-generation** — feedback loops, adaptive improvement, analytics

## License

MIT
