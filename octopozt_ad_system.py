"""
Octopozt Ad System — ComfyUI Custom Node
Generates production-ready NanaBanana2 prompts for ad generation.
Handles talent identity lock, product preservation, and brand style analysis.
"""

import os
import io
import base64
import json
import torch
import numpy as np
from PIL import Image

# ── Gemini SDK ────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[OctopoztAdSystem] google-generativeai not installed. Run: pip install google-generativeai")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to PIL."""
    arr = tensor[0].cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_base64(img: Image.Image, fmt="JPEG", quality=92) -> str:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG")
    else:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def gemini_call(api_key: str, model_name: str, system_prompt: str,
                user_prompt: str, images: list, max_tokens: int = 4096) -> str:
    """Single Gemini API call. images = list of PIL.Image."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    parts = []
    for img in images:
        parts.append(img)
    parts.append(user_prompt)
    response = model.generate_content(
        parts,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
    )
    return response.text.strip()


# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_ASSETS = """You are a professional casting director and product photographer.
Describe visual assets with EXTREME photographic precision for AI image generation.
Every physical detail is critical for faithful reproduction. Never guess — only describe what you see."""

SYSTEM_BRAND = """You are a professional creative director and visual strategist.
Describe brand visual style with extreme precision for AI image generation replication.
Focus ONLY on photographic technique, lighting, composition, and aesthetics.
Never mention demographics, diversity, or casting."""

SYSTEM_PROMPT_GENERATOR = """You are a world-class advertising photographer writing briefs for NanaBanana2 (Gemini image generation).

CRITICAL RULES — NEVER VIOLATE:
1. IDENTITY LOCK: The talent in the reference image is LOCKED. Describe them with 100% fidelity at the START of every prompt. Include every distinctive physical feature explicitly.
2. PRODUCT LOCK: The product must be reproduced exactly — same shape, same label, same colors, zero modifications.
3. TEXT LOCK: Ad copy text must appear EXACTLY as provided — no changes, no omissions.
4. Write in dense, natural photographic language. NO JSON. NO bullet points. NO numbered lists.
5. Every prompt MUST start with the identity opener and end with the logo/text rules."""

PROMPT_ASSETS = """Analyze the provided images and describe each asset with EXTREME precision.

Image 1 = TALENT. Start with "TALENT:". Describe: exact age range, ethnicity, skin tone (use specific descriptors like "warm medium olive"), hair (color, texture, length, style — flag ANY unique features like white/silver streaks with exact location), eyes (color, shape), facial structure, expression, clothing (every item, color, fabric), body language, any distinctive features (moles, piercings, tattoos, accessories).

Image 2 = PRODUCT. Start with "PRODUCT:". Describe: exact shape and proportions, all colors (primary and secondary), complete label text visible, typography style, material and finish.

Image 3 = BRAND LOGO. Start with "LOGO:". Describe: exact colors, typography (font style, weight, casing), any icons or symbols, gradients, overall shape.

Write as if briefing a photographer who has NEVER seen these assets and must reproduce them with 100% fidelity."""

PROMPT_BRAND = """Analyze the provided brand reference images and extract a PRECISE visual style guide.

Include:
1. COLOR PALETTE: Primary and accent colors, mood, contrast style
2. PHOTOGRAPHY STYLE: Camera angles, lighting type, depth of field, grain/texture
3. COMPOSITION: Layout patterns, product placement, negative space
4. PHOTOGRAPHIC APPROACH: Camera distance, expression style, body language, pose style — describe HOW subjects are photographed only, never WHO they are
5. TYPOGRAPHY MOOD: Bold/subtle, integrated/overlay style
6. OVERALL AESTHETIC: One dense paragraph describing the brand's visual DNA for AI replication

Output ONLY the structured guide. No preamble."""

def build_final_prompt(asset_desc: str, brand_style: str, copy_text: str,
                       objective: str, tone: str, num_variations: int) -> str:
    return f"""You have received:

=== ASSET DESCRIPTIONS ===
{asset_desc}

=== BRAND VISUAL STYLE GUIDE ===
{brand_style}

=== CAMPAIGN BRIEF ===
- Ad copy text (EXACT, never change): "{copy_text}"
- Campaign objective: {objective}
- Visual tone: {tone}

CRITICAL PRE-PROCESSING RULE: The brand style guide may mention "diversity" or similar casting language. COMPLETELY IGNORE all such references. The ONLY talent is the EXACT individual described in ASSET DESCRIPTIONS above.

Generate {num_variations} scene variation prompts for NanaBanana2.

MANDATORY STRUCTURE FOR EVERY PROMPT:
Start with: "The EXACT same [brief physical description of talent from ASSET DESCRIPTIONS — hair, skin, eyes, distinctive features]. IDENTITY LOCK: reproduce this person with 100% fidelity from the reference image. Do NOT generate a different person."
Then: cinematic scene description (location, lighting, camera angle, mood, what talent is doing with product)
End with: "Ad copy text '{copy_text}' integrated naturally into the scene — no background box, no banner, no graphic overlay. Brand logo in top right corner, clean and unmodified. Product is the EXACT [product name from description] — do not modify shape, label, or branding."

Use these {num_variations} distinct scenes (adapt to brand style):
1. Urban rooftop, golden hour warm sunlight
2. Beach boardwalk, sunset tones  
3. City alley at night, colorful neon lighting
4. Mountain trail overlook, bright midday expansive sky
5. Modern gym interior, natural window light
6. Outdoor cafe patio, relaxed daytime

Separate each prompt with *
Do NOT use JSON. Do NOT number the prompts. Start directly with the first prompt."""


# ── Main Node ─────────────────────────────────────────────────────────────────

class OctopoztAdSystem:
    """
    Octopozt Ad System
    Generates NanaBanana2 prompts with talent identity lock,
    product preservation, and brand style analysis.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("NANO_BANANA_PROMPT", "ASSET_DESCRIPTION", "BRAND_STYLE", "DEBUG_LOG")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "talent_image":   ("IMAGE",),
                "product_image":  ("IMAGE",),
                "brand_logo":     ("IMAGE",),
                "copy_text":      ("STRING", {"default": "ENERGÍA PARA TU DÍA A DÍA", "multiline": False}),
                "objective":      ("STRING", {"default": "Awareness", "multiline": False}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "tone": (["Energético", "Premium", "Cercano", "Urgente", "Inspiracional", "Divertido"],
                         {"default": "Energético"}),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                          {"default": "gemini-2.5-flash"}),
                "num_variations": ("INT", {"default": 6, "min": 1, "max": 10}),
                "brand_ref_1":    ("IMAGE",),
                "brand_ref_2":    ("IMAGE",),
                "brand_ref_3":    ("IMAGE",),
                "brand_ref_4":    ("IMAGE",),
                "brand_ref_5":    ("IMAGE",),
                "typography_ref": ("IMAGE",),
            }
        }

    def generate(self, talent_image, product_image, brand_logo, copy_text,
                 objective, gemini_api_key,
                 tone="Energético", model="gemini-2.5-flash", num_variations=6,
                 brand_ref_1=None, brand_ref_2=None, brand_ref_3=None,
                 brand_ref_4=None, brand_ref_5=None, typography_ref=None):

        debug_lines = []

        if not GEMINI_AVAILABLE:
            err = "[OctopoztAdSystem] ERROR: google-generativeai not installed."
            return (err, err, err, err)

        api_key = gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            err = "[OctopoztAdSystem] ERROR: No Gemini API key provided."
            return (err, err, err, err)

        # ── CALL 1: Analyze assets (talent, product, logo) ────────────────────
        debug_lines.append("=== CALL 1: Asset Analysis ===")
        try:
            asset_imgs = [
                tensor_to_pil(talent_image),
                tensor_to_pil(product_image),
                tensor_to_pil(brand_logo),
            ]
            asset_desc = gemini_call(
                api_key, model,
                SYSTEM_ASSETS,
                PROMPT_ASSETS,
                asset_imgs,
            )
            debug_lines.append(f"Assets analyzed: {len(asset_desc)} chars")
        except Exception as e:
            asset_desc = f"[Asset analysis failed: {e}]"
            debug_lines.append(f"CALL 1 ERROR: {e}")

        # ── CALL 2: Analyze brand style references ────────────────────────────
        brand_style = ""
        brand_refs = [r for r in [brand_ref_1, brand_ref_2, brand_ref_3, brand_ref_4, brand_ref_5] if r is not None]

        if brand_refs:
            debug_lines.append(f"=== CALL 2: Brand Style ({len(brand_refs)} refs) ===")
            try:
                ref_imgs = [tensor_to_pil(r) for r in brand_refs]
                brand_style = gemini_call(
                    api_key, model,
                    SYSTEM_BRAND,
                    PROMPT_BRAND,
                    ref_imgs,
                )
                debug_lines.append(f"Brand style analyzed: {len(brand_style)} chars")
            except Exception as e:
                brand_style = f"[Brand style analysis failed: {e}]"
                debug_lines.append(f"CALL 2 ERROR: {e}")
        else:
            brand_style = "No brand references provided. Use a bold, energetic, lifestyle photography style."
            debug_lines.append("CALL 2: Skipped (no brand refs)")

        # ── CALL 3: Generate final NanaBanana2 prompts ────────────────────────
        debug_lines.append(f"=== CALL 3: Prompt Generation ({num_variations} variations) ===")
        try:
            final_user_prompt = build_final_prompt(
                asset_desc, brand_style, copy_text, objective, tone, num_variations
            )
            nano_prompt = gemini_call(
                api_key, model,
                SYSTEM_PROMPT_GENERATOR,
                final_user_prompt,
                [],  # no images needed — asset descriptions are in text
                max_tokens=8192,
            )
            debug_lines.append(f"Prompt generated: {len(nano_prompt)} chars")
        except Exception as e:
            nano_prompt = f"[Prompt generation failed: {e}]"
            debug_lines.append(f"CALL 3 ERROR: {e}")

        debug_log = "\n".join(debug_lines)
        return (nano_prompt, asset_desc, brand_style, debug_log)


# ── Batch Images (no crop, no distortion) ────────────────────────────────────

class OctopoztBatchImages:
    """
    Batch multiple images preserving original aspect ratios.
    Pads to the largest image size instead of cropping.
    Up to 6 images.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "batch"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "pad_color": (["black", "white", "gray"], {"default": "black"}),
            }
        }

    def batch(self, image_1, image_2=None, image_3=None,
              image_4=None, image_5=None, image_6=None, pad_color="black"):

        tensors = [t for t in [image_1, image_2, image_3, image_4, image_5, image_6] if t is not None]

        # Find max H and W
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        c     = tensors[0].shape[3]

        pad_val = {"black": 0.0, "white": 1.0, "gray": 0.5}.get(pad_color, 0.0)

        padded = []
        for t in tensors:
            _, h, w, _ = t.shape
            if h == max_h and w == max_w:
                padded.append(t)
            else:
                canvas = torch.full((1, max_h, max_w, c), pad_val, dtype=t.dtype)
                canvas[0, :h, :w, :] = t[0]
                padded.append(canvas)

        return (torch.cat(padded, dim=0),)


# ── Registrations ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "OctopoztAdSystem":   OctopoztAdSystem,
    "OctopoztBatchImages": OctopoztBatchImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OctopoztAdSystem":    "🐙 Octopozt Ad System",
    "OctopoztBatchImages": "🐙 Octopozt Batch Images",
}
