"""
Octopozt Suno Music — ComfyUI Custom Node
Genera música real desde texto usando Suno AI.
Output: AUDIO compatible con OctopoztAutoMix.
"""

import io
import time
import requests
import numpy as np
import torch
import soundfile as sf


def mp3_bytes_to_tensor(mp3_data: bytes):
    """Convierte bytes MP3 a tensor ComfyUI usando soundfile."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_data)
        tmp = f.name
    try:
        data, sr = sf.read(tmp, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).unsqueeze(0)  # (1, ch, samples)
        return waveform, sr
    finally:
        os.unlink(tmp)


class OctopoztSunoMusic:
    """
    Genera música de fondo usando Suno AI.
    Devuelve AUDIO directo para conectar al OctopoztAutoMix.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("AUDIO", "URL", "INFO")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "suno_cookie": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "Cookie __client de auth.suno.com"}),
                "prompt":      ("STRING", {"default": "Upbeat latin pop instrumental for a product ad, energetic and modern, no vocals",
                                           "multiline": True,
                                           "tooltip": "Describe la música que quieres"}),
                "tags":        ("STRING", {"default": "latin pop, upbeat, commercial, instrumental",
                                           "multiline": False,
                                           "tooltip": "Géneros y estilos separados por coma"}),
                "instrumental": ("BOOLEAN", {"default": True,
                                             "label_on": "Sin letra (instrumental)",
                                             "label_off": "Con letra"}),
                "song_index":  (["1", "2"], {"default": "1",
                                              "tooltip": "Suno genera 2 variaciones — elige cuál usar"}),
            }
        }

    def generate(self, suno_cookie, prompt, tags, instrumental, song_index):
        if not suno_cookie.strip():
            raise ValueError("Falta la cookie de Suno")

        cookie_str = suno_cookie.strip()

        session = requests.Session()
        session.headers.update({
            "Cookie":     cookie_str,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Origin":     "https://suno.com",
            "Referer":    "https://suno.com/",
        })

        # ── 1. Obtener JWT token ───────────────────────────────────────────────
        jwt_resp = session.get("https://studio-api.suno.ai/api/auth/token/", timeout=15)
        if jwt_resp.status_code != 200:
            raise RuntimeError(f"Error obteniendo token: {jwt_resp.status_code} — {jwt_resp.text[:200]}")

        jwt = jwt_resp.json().get("token")
        if not jwt:
            raise RuntimeError("No se obtuvo JWT — la cookie puede estar expirada")

        session.headers["Authorization"] = f"Bearer {jwt}"

        # ── 2. Generar música ──────────────────────────────────────────────────
        gen_resp = session.post(
            "https://studio-api.suno.ai/api/generate/v2/",
            json={
                "gpt_description_prompt": prompt,
                "tags":                   tags if not instrumental else tags,
                "mv":                     "chirp-v3-5",
                "prompt":                 "",
                "make_instrumental":      instrumental,
                "generation_type":        "TEXT",
            },
            timeout=30,
        )

        if gen_resp.status_code != 200:
            raise RuntimeError(f"Error generando en Suno: {gen_resp.status_code} — {gen_resp.text[:200]}")

        clips = gen_resp.json().get("clips", [])
        if not clips:
            raise RuntimeError("Suno no devolvió clips")

        clip_ids = [c["id"] for c in clips]
        idx = int(song_index) - 1
        target_id = clip_ids[min(idx, len(clip_ids)-1)]

        # ── 3. Esperar que esté listo (polling) ───────────────────────────────
        audio_url = None
        for attempt in range(30):  # máx ~60 segundos
            time.sleep(2)
            status_resp = session.get(
                f"https://studio-api.suno.ai/api/feed/?ids={target_id}",
                timeout=15,
            )
            if status_resp.status_code != 200:
                continue
            feed = status_resp.json()
            clips_status = feed if isinstance(feed, list) else feed.get("clips", [])
            for clip in clips_status:
                if clip.get("id") == target_id:
                    status = clip.get("status", "")
                    if status == "complete":
                        audio_url = clip.get("audio_url")
                        break
                    elif status == "error":
                        raise RuntimeError(f"Suno reportó error en la generación")
            if audio_url:
                break

        if not audio_url:
            raise RuntimeError("Timeout esperando Suno — intenta de nuevo")

        # ── 4. Descargar MP3 ──────────────────────────────────────────────────
        mp3_resp = requests.get(audio_url, timeout=60)
        if mp3_resp.status_code != 200:
            raise RuntimeError(f"Error descargando audio: {mp3_resp.status_code}")

        # ── 5. Convertir a tensor ─────────────────────────────────────────────
        try:
            waveform, sample_rate = mp3_bytes_to_tensor(mp3_resp.content)
        except Exception as e:
            raise RuntimeError(f"Error decodificando MP3: {e}")

        output_audio = {"waveform": waveform, "sample_rate": sample_rate}

        info = (
            f"=== SUNO MUSIC ===\n"
            f"Prompt: {prompt[:80]}\n"
            f"Tags: {tags}\n"
            f"Instrumental: {instrumental}\n"
            f"Clip ID: {target_id}\n"
            f"Sample rate: {sample_rate} Hz | Canales: {waveform.shape[1]}\n"
            f"Duración: {waveform.shape[2]/sample_rate:.1f}s"
        )

        return (output_audio, audio_url, info)


NODE_CLASS_MAPPINGS_SUNO = {
    "OctopoztSunoMusic": OctopoztSunoMusic,
}

NODE_DISPLAY_NAME_MAPPINGS_SUNO = {
    "OctopoztSunoMusic": "🐙 Octopozt Suno Music",
}
