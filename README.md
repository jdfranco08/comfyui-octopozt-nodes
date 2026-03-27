# 🐙 ComfyUI Octopozt Nodes

Custom nodes for AI ad generation with identity lock, product preservation, and brand style analysis.

## Nodes

### 🐙 Octopozt Ad System
Generates production-ready NanaBanana2 prompts in 3 Gemini calls:
- **Call 1:** Analyzes talent, product, and logo with extreme precision
- **Call 2:** Extracts brand visual style from reference images
- **Call 3:** Generates N scene variation prompts in natural language (no JSON)

**Inputs:**
- `talent_image` — the person/character
- `product_image` — product photo
- `brand_logo` — brand logo (any size/aspect ratio, no distortion)
- `brand_ref_1..5` *(optional)* — existing brand ads for style reference
- `typography_ref` *(optional)* — typography style reference
- `copy_text` — ad copy text (appears exactly as written)
- `objective` — Awareness / Conversión / Engagement / Tráfico / Leads
- `tone` — Energético / Premium / Cercano / Urgente / Inspiracional / Divertido
- `num_variations` — 1–10 scene variations
- `gemini_api_key` — or set `GEMINI_API_KEY` env variable
- `model` — gemini-2.5-flash (default) / gemini-2.5-pro / gemini-2.0-flash

**Outputs:**
- `NANO_BANANA_PROMPT` — ready to connect to NanaBanana2
- `ASSET_DESCRIPTION` — talent + product + logo description (for debug)
- `BRAND_STYLE` — extracted brand style guide (for debug)
- `DEBUG_LOG` — call logs

---

### 🐙 Octopozt Batch Images
Batches up to 6 images **without cropping or distorting** — pads to the largest size instead.

**Inputs:** image_1..6, pad_color (black/white/gray)  
**Output:** batched IMAGE tensor

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jdfranco08/comfyui-octopozt-nodes
pip install google-generativeai
```

## Requirements
```
google-generativeai>=0.8.0
torch
Pillow
numpy
```
