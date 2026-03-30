"""
Octopozt ElevenLabs Music — ComfyUI Custom Node
Genera música desde texto usando la API de ElevenLabs.
Output: AUDIO compatible con OctopoztAutoMix y VideoHelperSuite.
"""

import io
import requests
import numpy as np
import torch


class OctopoztElevenMusic:
    """
    Genera música de fondo desde un prompt de texto via ElevenLabs.
    Conecta el output directo al OctopoztAutoMix.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "INFO")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Tu API key de ElevenLabs"}),
                "prompt":     ("STRING", {"default": "Upbeat latin pop background music for a product ad, energetic, modern",
                                          "multiline": True,
                                          "tooltip": "Describe la música que quieres generar"}),
                "duration_s": ("INT",    {"default": 30, "min": 5, "max": 120, "step": 5,
                                          "tooltip": "Duración en segundos (5-120)"}),
                "influence":  ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "display": "slider",
                                          "tooltip": "Qué tanto sigue el prompt (0=libre, 1=estricto)"}),
            }
        }

    def generate(self, api_key, prompt, duration_s, influence):
        if not api_key.strip():
            raise ValueError("Falta la API key de ElevenLabs")

        url = "https://api.elevenlabs.io/v1/sound-generation"
        headers = {
            "xi-api-key": api_key.strip(),
            "Content-Type": "application/json",
        }
        payload = {
            "text":               prompt,
            "duration_seconds":   duration_s,
            "prompt_influence":   influence,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs error {response.status_code}: {response.text[:300]}")

        # ── Decodificar MP3 → tensor ───────────────────────────────────────────
        try:
            import torchaudio
            audio_bytes = io.BytesIO(response.content)
            waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")
        except Exception as e:
            raise RuntimeError(f"Error decodificando audio: {e}")

        # Normalizar a (1, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        output_audio = {
            "waveform":    waveform,
            "sample_rate": sample_rate,
        }

        info = (
            f"=== ELEVEN MUSIC ===\n"
            f"Prompt: {prompt[:80]}...\n"
            f"Duración: {duration_s}s\n"
            f"Influence: {influence}\n"
            f"Sample rate: {sample_rate} Hz\n"
            f"Canales: {waveform.shape[1]}\n"
            f"Samples: {waveform.shape[2]}"
        )

        return (output_audio, info)


class OctopoztElevenTTS:
    """
    Genera voz en off desde texto via ElevenLabs.
    Conecta el output directo al OctopoztAutoMix como 'voice'.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "INFO")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Tu API key de ElevenLabs"}),
                "text":       ("STRING", {"default": "Descubre nuestra nueva colección. Calidad que se siente.",
                                          "multiline": True,
                                          "tooltip": "Script del voice over"}),
                "voice_id":   ("STRING", {"default": "21m00Tcm4TlvDq8ikWAM",
                                          "multiline": False,
                                          "tooltip": "ID de voz de ElevenLabs (Rachel por defecto)"}),
                "model":      (["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5"],
                                {"default": "eleven_multilingual_v2"}),
                "stability":  ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "display": "slider"}),
                "similarity":  ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "display": "slider"}),
            }
        }

    def generate(self, api_key, text, voice_id, model, stability, similarity):
        if not api_key.strip():
            raise ValueError("Falta la API key de ElevenLabs")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id.strip()}"
        headers = {
            "xi-api-key": api_key.strip(),
            "Content-Type": "application/json",
        }
        payload = {
            "text":       text,
            "model_id":   model,
            "voice_settings": {
                "stability":         stability,
                "similarity_boost":  similarity,
            },
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs error {response.status_code}: {response.text[:300]}")

        try:
            import torchaudio
            audio_bytes = io.BytesIO(response.content)
            waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")
        except Exception as e:
            raise RuntimeError(f"Error decodificando audio: {e}")

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        output_audio = {
            "waveform":    waveform,
            "sample_rate": sample_rate,
        }

        info = (
            f"=== ELEVEN TTS ===\n"
            f"Texto: {text[:80]}...\n"
            f"Voice ID: {voice_id}\n"
            f"Modelo: {model}\n"
            f"Sample rate: {sample_rate} Hz"
        )

        return (output_audio, info)


NODE_CLASS_MAPPINGS_ELEVEN = {
    "OctopoztElevenMusic": OctopoztElevenMusic,
    "OctopoztElevenTTS":   OctopoztElevenTTS,
}

NODE_DISPLAY_NAME_MAPPINGS_ELEVEN = {
    "OctopoztElevenMusic": "🐙 Octopozt ElevenLabs Music",
    "OctopoztElevenTTS":   "🐙 Octopozt ElevenLabs TTS",
}
