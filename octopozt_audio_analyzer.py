"""
Octopozt Audio Analyzer — ComfyUI Custom Node
Analiza dos pistas de audio y calcula automáticamente
los dB óptimos para que la voz quede siempre por encima de la música.
"""

import torch
import numpy as np


def waveform_to_mono(waveform):
    """Convierte tensor (batch, channels, samples) a numpy mono float32."""
    return waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)


def rms_db(signal):
    """Calcula el RMS en dB de una señal. Retorna -inf si silencio."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-9:
        return -96.0
    return float(20 * np.log10(rms))


def peak_db(signal):
    """Calcula el pico en dB."""
    peak = np.abs(signal).max()
    if peak < 1e-9:
        return -96.0
    return float(20 * np.log10(peak))


class OctopoztAudioAnalyzer:
    """
    Analiza voz y música, calcula automáticamente:
    - voice_db: cuánto ajustar la voz para que esté al nivel deseado
    - music_db: cuánto bajar la música para que la voz quede X dB por encima
    - duck_db:  nivel de ducking durante la voz

    Conecta los outputs directamente al OctopoztAudioMixer.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "analyze"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("voice_db", "music_db", "duck_db", "REPORT")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice":              ("AUDIO",),
                "music":              ("AUDIO",),
                "voice_target_db":    ("FLOAT", {"default": -6.0,  "min": -30.0, "max": 0.0,  "step": 0.5,
                                                 "display": "slider",
                                                 "tooltip": "Nivel RMS objetivo para la voz (recomendado: -6 dB)"}),
                "voice_above_music":  ("FLOAT", {"default":  8.0,  "min":  3.0,  "max": 30.0, "step": 0.5,
                                                 "display": "slider",
                                                 "tooltip": "Cuántos dB debe estar la voz por encima de la música"}),
                "duck_extra_db":      ("FLOAT", {"default": 10.0,  "min":  0.0,  "max": 30.0, "step": 0.5,
                                                 "display": "slider",
                                                 "tooltip": "dB extra para bajar la música durante la voz (ducking adicional)"}),
            }
        }

    def analyze(self, voice, music, voice_target_db, voice_above_music, duck_extra_db):

        v_np = waveform_to_mono(voice["waveform"])
        m_np = waveform_to_mono(music["waveform"])

        # ── Medir niveles actuales ─────────────────────────────────────────────
        v_rms_db  = rms_db(v_np)
        v_peak_db = peak_db(v_np)
        m_rms_db  = rms_db(m_np)
        m_peak_db = peak_db(m_np)

        # ── Calcular ajustes ───────────────────────────────────────────────────
        # 1. voice_db: cuánto subir/bajar la voz para llegar al target
        voice_db = voice_target_db - v_rms_db
        voice_db = max(-20.0, min(12.0, voice_db))  # clamp seguro

        # 2. music_db: bajar la música para que quede X dB por debajo de la voz
        #    voz objetivo = voice_target_db
        #    música objetivo = voice_target_db - voice_above_music
        music_target = voice_target_db - voice_above_music
        music_db = music_target - m_rms_db
        music_db = max(-40.0, min(6.0, music_db))

        # 3. duck_db: durante la voz, bajar aún más la música
        duck_db = music_db - duck_extra_db
        duck_db = max(-60.0, min(0.0, duck_db))

        # ── Reporte legible ────────────────────────────────────────────────────
        report = (
            f"=== ANÁLISIS DE AUDIO ===\n"
            f"\nVOZ:\n"
            f"  RMS actual:  {v_rms_db:+.1f} dB\n"
            f"  Pico actual: {v_peak_db:+.1f} dB\n"
            f"  Ajuste:      {voice_db:+.1f} dB\n"
            f"  RMS final:   {v_rms_db + voice_db:+.1f} dB\n"
            f"\nMÚSICA:\n"
            f"  RMS actual:  {m_rms_db:+.1f} dB\n"
            f"  Pico actual: {m_peak_db:+.1f} dB\n"
            f"  Ajuste:      {music_db:+.1f} dB\n"
            f"  RMS final:   {m_rms_db + music_db:+.1f} dB\n"
            f"\nDUCKING:\n"
            f"  Música durante voz: {duck_db:+.1f} dB\n"
            f"\nBALANCE:\n"
            f"  Voz sobre música: {(v_rms_db + voice_db) - (m_rms_db + music_db):+.1f} dB\n"
            f"  (objetivo: +{voice_above_music:.0f} dB)"
        )

        return (voice_db, music_db, duck_db, report)


NODE_CLASS_MAPPINGS_ANALYZER = {
    "OctopoztAudioAnalyzer": OctopoztAudioAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS_ANALYZER = {
    "OctopoztAudioAnalyzer": "🐙 Octopozt Audio Analyzer",
}
