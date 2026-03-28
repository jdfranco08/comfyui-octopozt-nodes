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
4. STYLE LOCK: Extract concrete photographic specs from the brand style guide (camera angle, lens, lighting, composition) and mandate them explicitly in EVERY prompt. If the brand uses low-angle shots — every variation must be low-angle. If it uses golden hour — use golden hour. Never default to generic eye-level flat shots.
5. Write in dense, natural photographic language. NO JSON. NO bullet points. NO numbered lists.
6. Every prompt MUST start with the identity opener, include explicit camera/lighting specs from brand style, and end with the logo/text rules.

TEXT OVERLAY RULE — ABSOLUTE AND NON-NEGOTIABLE:
The ad copy text must appear as a CLEAN TYPOGRAPHIC OVERLAY directly superimposed on top of the finished image — exactly like post-production subtitles or a floating caption added in Adobe Premiere or After Effects. Bold, legible, high contrast, NO background box, NO shadow box, NO banner, NO ribbon. The text floats clean above the scene. It is NEVER painted on walls, floors, buildings, bottles, clothing, or any physical surface in the scene. It is NOT part of the photography. It is a graphic layer placed on top. If you generate a scene where the text appears physically integrated into the environment, that is a critical failure."""

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
                       objective: str, tone: str, num_variations: int,
                       creative_brief: str = "", brand_context: str = "") -> str:
    brief_section = f"\n- Creative direction: {creative_brief}" if creative_brief.strip() else ""
    brand_ctx_section = f"\n- Brand personality: {brand_context}" if brand_context.strip() else ""

    return f"""You are generating a production brief JSON for NanaBanana2 image generation (Gemini).

STEP 1 — Extract from the BRAND VISUAL STYLE GUIDE these exact specs and use them in EVERY variation:
- Dominant camera angle (e.g. low-angle, worm's-eye, eye-level)
- Lighting type (e.g. harsh direct sunlight, golden hour, neon)
- Lens (e.g. 85mm f/1.4 portrait, 35mm wide)
- Composition pattern (e.g. subject in lower third, sky as negative space)

STEP 2 — Generate {num_variations} production brief JSONs, one per scene variation.

ABSOLUTE RULES FOR ALL VARIATIONS:
- talent: IDENTITY LOCK — same exact person from image_1. 100% fidelity. Same face, hair (note white streak if present), skin tone. NEVER generate a different person.
- product: use EXACT product from image_2. preserve_product_shape=true. do_not_modify_branding=true.
- text_overlay: The copy text must appear as a CLEAN CAPTION SUPERIMPOSED over the image — like post-production subtitles or a floating text overlay. NOT painted on walls, NOT carved into surfaces, NOT integrated into the scene environment. It floats cleanly above everything, bold and legible, positioned in the lower third or upper area with clear contrast against the background.
- logo: use EXACT logo from image_4. top right corner. do_not_redesign=true.
- Apply brand photographic style extracted in STEP 1 to EVERY variation.

=== ASSET DESCRIPTIONS ===
{asset_desc}

=== BRAND VISUAL STYLE GUIDE ===
{brand_style}

=== CAMPAIGN BRIEF ===
- Ad copy text (EXACT, never change): "{copy_text}"
- Campaign objective: {objective}
- Visual tone: {tone}{brief_section}{brand_ctx_section}

IGNORE any diversity/demographic language in the style guide. The ONLY talent is image_1.

Output exactly {num_variations} JSON objects separated by *
No markdown. No explanations. Start directly with {{

FORMAT per variation:
{{"scene_description":"...","composition":{{"camera":"LOW-ANGLE or exact angle from brand style","lens":"85mm f/1.4 or brand lens","focus":"...","lighting":"exact brand lighting type","depth":"shallow/medium"}},"style":{{"render":"ultra-realistic commercial photography, Sony A7R V","mood":"...","color_palette":"consistent with brand style guide"}},"talent":{{"reference_image":"image_1","instruction":"ABSOLUTE IDENTITY LOCK: Reproduce EXACT person from image_1. 100% fidelity. Same face, hair including white streak, skin tone. Do NOT generate a different person.","action":"...","expression":"..."}},"product":{{"reference_image":"image_2","placement":"...","enhancement":"condensation, glow, etc","rules":{{"preserve_product_shape":true,"do_not_modify_branding":true}}}},"text_overlay":{{"content":"{copy_text}","style":"CLEAN SUPERIMPOSED CAPTION — bold white text with subtle drop shadow, floating above the scene in post-production style. NOT painted on surfaces. NOT integrated into environment.","placement":"lower third, horizontally centered","constraints":{{"no_surface_integration":true,"no_environment_painting":true,"floating_caption_style":true}}}},"logo":{{"reference_image":"image_4","placement":"top right corner","style":"clean sharp no distortion","protection":{{"do_not_redesign":true,"keep_exact_colors":true}}}}}}"""


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
                "copy_text":      ("STRING", {"forceInput": True}),
                "objective":      ("STRING", {"forceInput": True}),
                "creative_brief": ("STRING", {"forceInput": True}),
                "brand_context":  ("STRING", {"forceInput": True}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "tone":           ("STRING", {"forceInput": True}),
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
                 objective, creative_brief, brand_context, gemini_api_key,
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
                asset_desc, brand_style, copy_text, objective, tone, num_variations,
                creative_brief=creative_brief, brand_context=brand_context
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

        # ── Ensure exactly num_variations prompts ─────────────────────────────
        parts = [p.strip() for p in nano_prompt.split("*") if p.strip()]
        debug_lines.append(f"Variations received: {len(parts)} / {num_variations} requested")

        if len(parts) < num_variations:
            # Build a simple fallback prompt using asset_desc + copy_text
            fallback = (
                f"The EXACT same person from the reference image. IDENTITY LOCK: same face, hair, skin tone — zero changes. "
                f"Standing in a clean modern lifestyle setting, holding the product naturally. "
                f"Bright natural lighting, medium shot, shallow depth of field. "
                f"IMPORTANT: Ad copy text '{copy_text}' must appear as a CLEAN SUPERIMPOSED CAPTION floating above the scene — "
                f"NOT painted on surfaces, NOT integrated into the environment. Post-production text overlay style only. "
                f"Brand logo top right corner, clean and unmodified."
            )
            while len(parts) < num_variations:
                parts.append(fallback)
                debug_lines.append(f"Fallback prompt injected for variation {len(parts)}")

        # Trim if too many
        parts = parts[:num_variations]
        nano_prompt = " * ".join(parts)

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

        # Normalize all tensors to RGB (3 channels) — drop alpha if present
        def to_rgb(t):
            if t.shape[3] == 4:
                # Composite RGBA onto pad_color background
                rgb = t[..., :3]
                alpha = t[..., 3:4]
                bg = torch.full_like(rgb, pad_val)
                return rgb * alpha + bg * (1.0 - alpha)
            return t

        tensors = [to_rgb(t) for t in tensors]
        c = 3  # always RGB after normalization

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
