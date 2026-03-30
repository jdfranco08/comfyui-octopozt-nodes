"""
Octopozt MusicGen — ComfyUI Custom Node
Genera música local usando Meta MusicGen via HuggingFace Transformers.
Sin API key — corre 100% local en Apple Silicon MPS o CPU.
"""

import torch
import numpy as np


class OctopoztMusicGen:
    """
    Genera música instrumental usando MusicGen de Meta via transformers.
    Sin API key — corre 100% local.
    Primera vez descarga el modelo (~1.5 GB), luego queda cacheado.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "INFO")

    _pipeline_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":     ("STRING", {"default": "upbeat latin pop instrumental for a product ad, energetic, modern, no vocals",
                                           "multiline": True}),
                "model":      (["small", "medium"], {"default": "small",
                                "tooltip": "small=~300MB rápido, medium=~1.5GB mejor calidad"}),
                "duration_s": ("INT",   {"default": 15, "min": 5, "max": 30, "step": 5,
                                          "tooltip": "Duración en segundos"}),
                "seed":       ("INT",   {"default": 0, "min": 0, "max": 99999,
                                          "tooltip": "0 = aleatorio"}),
            }
        }

    def generate(self, prompt, model, duration_s, seed):
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError("Instala transformers: pip install transformers accelerate")

        model_id = f"facebook/musicgen-{model}"

        # ── Device ────────────────────────────────────────────────────────────
        # MusicGen tiene bug en MPS (conv1d > 65536 canales) — usar CPU en Mac
        # En Linux/Windows con CUDA sí usa GPU
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"  # CPU en Mac — funciona bien con Apple Silicon

        # ── Cargar pipeline (cacheado) ─────────────────────────────────────────
        if model_id not in OctopoztMusicGen._pipeline_cache:
            print(f"[MusicGen] Cargando {model_id} en {device}...")
            synthesiser = pipeline(
                "text-to-audio",
                model=model_id,
                device=device,
            )
            OctopoztMusicGen._pipeline_cache[model_id] = synthesiser
        else:
            synthesiser = OctopoztMusicGen._pipeline_cache[model_id]

        # ── Seed ──────────────────────────────────────────────────────────────
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ── Generar ───────────────────────────────────────────────────────────
        print(f"[MusicGen] Generando {duration_s}s...")
        result = synthesiser(
            prompt,
            forward_params={"max_new_tokens": int(256 * duration_s / 5)},
        )

        audio_array = result["audio"]   # numpy array
        sample_rate = result["sampling_rate"]

        # ── Convertir a tensor ComfyUI (1, channels, samples) ─────────────────
        if audio_array.ndim == 1:
            audio_array = audio_array[np.newaxis, :]  # (1, samples)
        elif audio_array.ndim == 2 and audio_array.shape[0] > audio_array.shape[1]:
            audio_array = audio_array.T  # (channels, samples)

        waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)  # (1, ch, samples)

        output_audio = {"waveform": waveform, "sample_rate": sample_rate}

        duration_actual = waveform.shape[2] / sample_rate
        info = (
            f"=== MUSICGEN LOCAL ===\n"
            f"Modelo: {model_id}\n"
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
