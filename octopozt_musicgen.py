"""
Octopozt MusicGen — ComfyUI Custom Node
Genera música local usando Meta MusicGen (sin API key).
Corre en Apple Silicon MPS o CPU.
Output: AUDIO compatible con OctopoztAutoMix.
"""

import torch
import numpy as np


class OctopoztMusicGen:
    """
    Genera música instrumental usando MusicGen de Meta.
    Sin API key — corre 100% local en tu Mac.
    Primera vez descarga el modelo (~1.5 GB), luego queda cacheado.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "INFO")

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":    ("STRING", {"default": "upbeat latin pop instrumental for a product ad, energetic, modern, no vocals",
                                         "multiline": True,
                                         "tooltip": "Describe la música que quieres generar"}),
                "model":     (["small", "medium"], {"default": "small",
                               "tooltip": "small=~300MB rápido, medium=~1.5GB mejor calidad"}),
                "duration_s": ("INT",  {"default": 15, "min": 5, "max": 60, "step": 5,
                                        "tooltip": "Duración en segundos (más tiempo = más lento)"}),
                "seed":       ("INT",  {"default": 0, "min": 0, "max": 99999,
                                        "tooltip": "0 = aleatorio cada vez"}),
            }
        }

    def generate(self, prompt, model, duration_s, seed):
        try:
            from audiocraft.models import MusicGen
            from audiocraft.data.audio import audio_write
        except ImportError:
            raise RuntimeError(
                "audiocraft no está instalado. Ejecuta:\n"
                "~/Documents/ComfyUI/.venv/bin/python3 -m pip install audiocraft"
            )

        # ── Cargar modelo (cacheado en memoria) ───────────────────────────────
        model_key = f"facebook/musicgen-{model}"
        if model_key not in OctopoztMusicGen._model_cache:
            print(f"[OctopoztMusicGen] Cargando {model_key}...")
            mg = MusicGen.get_pretrained(model_key)
            OctopoztMusicGen._model_cache[model_key] = mg
        else:
            mg = OctopoztMusicGen._model_cache[model_key]

        # ── Configurar device ─────────────────────────────────────────────────
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        mg.set_generation_params(duration=duration_s)

        if seed > 0:
            torch.manual_seed(seed)

        # ── Generar ───────────────────────────────────────────────────────────
        print(f"[OctopoztMusicGen] Generando {duration_s}s en {device}...")
        with torch.no_grad():
            wav = mg.generate([prompt])  # shape: (1, 1, samples)

        # ── Convertir a formato ComfyUI ───────────────────────────────────────
        sample_rate = mg.sample_rate
        audio_tensor = wav[0].cpu()  # (1, samples) o (channels, samples)

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)

        waveform = audio_tensor.unsqueeze(0)  # (1, channels, samples)

        output_audio = {"waveform": waveform, "sample_rate": sample_rate}

        duration_actual = waveform.shape[2] / sample_rate
        info = (
            f"=== MUSICGEN LOCAL ===\n"
            f"Modelo: {model_key}\n"
            f"Device: {device}\n"
            f"Prompt: {prompt[:80]}\n"
            f"Duración: {duration_actual:.1f}s\n"
            f"Sample rate: {sample_rate} Hz"
        )

        return (output_audio, info)


NODE_CLASS_MAPPINGS_MUSICGEN = {
    "OctopoztMusicGen": OctopoztMusicGen,
}

NODE_DISPLAY_NAME_MAPPINGS_MUSICGEN = {
    "OctopoztMusicGen": "🐙 Octopozt MusicGen (local)",
}
