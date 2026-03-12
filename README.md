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

```
app.py          Streamlit UI — chat interface, sidebar controls, generation flow
agent.py        GPT-5-mini agent — conversation, field extraction, brief parsing
prompts.py      System prompt — pipeline rules, brief format, behavior
generate.py     Image generation backends — OpenAI and Gemini APIs
schema.py       Field definitions, style presets, validation
```

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- (Optional) A Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Install & Run

```bash
git clone https://github.com/zikunye2/ad-campaign-agent.git
cd ad-campaign-agent
pip install -r requirements.txt
streamlit run app.py
```

Enter your API key(s) in the sidebar and start chatting.

### API Key Safety

API keys are entered in the browser and stored **only in your session memory**. They are never saved to disk, logged, or sent anywhere except to the OpenAI/Google APIs. When you close the tab, keys are gone.

## Models

| Model | API | Style Reference | Notes |
|-------|-----|-----------------|-------|
| **GPT Image 1.5** | OpenAI | Text description only | 3 quality tiers (low/medium/high), configurable resolution |
| **Gemini 2.5 Flash** | Google | Image + text (native) | Multimodal LLM, uses style image directly |

## For Students

This repo is designed for you to fork and extend. Some ideas:

- Add new style presets (drop a PNG in `style/`, add to `STYLE_PRESETS` in `schema.py`)
- Add a new image generation backend (e.g. Stability AI, Flux)
- Add image editing / refinement after generation
- Add multi-image generation (A/B test variants)
- Add campaign analytics or audience insights
- Improve the creative brief format

## License

MIT
