"""
Octopozt ElevenLabs Music + TTS — ComfyUI Custom Node
Genera música y voz desde texto usando la API de ElevenLabs.
Pide PCM/WAV para evitar dependencia de ffprobe/ffmpeg.
"""

import io
import requests
import numpy as np
import torch
import soundfile as sf


def pcm_to_tensor(raw_bytes, sample_rate=44100, channels=1, bit_depth=16):
    """Convierte bytes PCM raw a tensor ComfyUI."""
    dtype = np.int16 if bit_depth == 16 else np.int32
    samples = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
    samples /= float(2 ** (bit_depth - 1))
    if channels == 2:
        samples = samples.reshape(-1, 2).T  # (2, samples)
    else:
        samples = samples.reshape(1, -1)    # (1, samples)
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, ch, samples)
    return waveform, sample_rate


def wav_bytes_to_tensor(wav_bytes):
    """Convierte bytes WAV a tensor ComfyUI usando soundfile."""
    buf = io.BytesIO(wav_bytes)
    data, sample_rate = sf.read(buf, dtype="float32", always_2d=True)
    # data shape: (samples, channels) → necesitamos (1, channels, samples)
    waveform = torch.from_numpy(data.T).unsqueeze(0)
    return waveform, sample_rate


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

        # Pedir PCM 44100Hz mono para evitar ffprobe
        url = "https://api.elevenlabs.io/v1/sound-generation"
        headers = {
            "xi-api-key":  api_key.strip(),
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",  # ElevenLabs Music solo devuelve mp3
        }
        payload = {
            "text":             prompt,
            "duration_seconds": duration_s,
            "prompt_influence": influence,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs error {response.status_code}: {response.text[:300]}")

        # Guardar como archivo temporal y leer con soundfile via mpg123 si está disponible
        # Si no, usar el truco de wav container
        try:
            # Intentar leer el MP3 con soundfile (requiere libsndfile con mp3)
            waveform, sample_rate = wav_bytes_to_tensor(response.content)
        except Exception:
            # Fallback: reencapsular como WAV usando numpy directo
            # ElevenLabs devuelve MP3 — escribirlo a disco temp y leer
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                tmp_path = f.name
            try:
                data, sample_rate = sf.read(tmp_path, dtype="float32", always_2d=True)
                waveform = torch.from_numpy(data.T).unsqueeze(0)
            finally:
                os.unlink(tmp_path)

        output_audio = {"waveform": waveform, "sample_rate": sample_rate}

        info = (
            f"=== ELEVEN MUSIC ===\n"
            f"Prompt: {prompt[:80]}\n"
            f"Duración: {duration_s}s | Influence: {influence}\n"
            f"Sample rate: {sample_rate} Hz | Canales: {waveform.shape[1]}"
        )

        return (output_audio, info)


class OctopoztElevenTTS:
    """
    Genera voz en off desde texto via ElevenLabs.
    Conecta el output al OctopoztAutoMix como 'voice'.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "INFO")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":   ("STRING", {"default": "", "multiline": False}),
                "text":      ("STRING", {"default": "Descubre nuestra nueva colección. Calidad que se siente.",
                                         "multiline": True}),
                "voice_id":  ("STRING", {"default": "21m00Tcm4TlvDq8ikWAM", "multiline": False,
                                          "tooltip": "ID de voz de ElevenLabs (Rachel por defecto)"}),
                "model":     (["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5"],
                               {"default": "eleven_multilingual_v2"}),
                "stability": ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "display": "slider"}),
                "similarity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "display": "slider"}),
            }
        }

    def generate(self, api_key, text, voice_id, model, stability, similarity):
        if not api_key.strip():
            raise ValueError("Falta la API key de ElevenLabs")

        # Pedir PCM WAV directamente — sin ffprobe
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id.strip()}/stream"
        headers = {
            "xi-api-key":   api_key.strip(),
            "Content-Type": "application/json",
            "Accept":       "audio/wav",  # WAV → soundfile lo lee sin ffprobe
        }
        payload = {
            "text":     text,
            "model_id": model,
            "voice_settings": {
                "stability":        stability,
                "similarity_boost": similarity,
            },
            "output_format": "pcm_44100",  # PCM raw 44100Hz
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS error {response.status_code}: {response.text[:300]}")

        # PCM raw 44100 mono 16bit → tensor
        try:
            waveform, sample_rate = pcm_to_tensor(response.content, sample_rate=44100, channels=1, bit_depth=16)
        except Exception as e:
            raise RuntimeError(f"Error decodificando PCM: {e}")

        output_audio = {"waveform": waveform, "sample_rate": sample_rate}

        info = (
            f"=== ELEVEN TTS ===\n"
            f"Texto: {text[:80]}\n"
            f"Voice: {voice_id} | Modelo: {model}\n"
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
