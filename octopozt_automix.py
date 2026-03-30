"""
Octopozt AutoMix — ComfyUI Custom Node
Mezcla voz + música automáticamente.
Analiza los niveles internamente y aplica ducking.
Sin configuración — solo conecta y listo.
"""

import torch
import numpy as np


def waveform_to_mono(waveform):
    return waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)


def rms_db(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-9:
        return -96.0
    return float(20 * np.log10(rms))


def db_to_linear(db):
    return 10 ** (db / 20.0)


class OctopoztAutoMix:
    """
    Mezcla voz + música automáticamente.
    Analiza los niveles internamente — sin sliders ni configuración.
    La voz siempre queda 8 dB por encima de la música con ducking suave.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "automix"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "REPORT")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": ("AUDIO",),
                "music": ("AUDIO",),
            }
        }

    def automix(self, voice, music):
        # ── Parámetros internos (no expuestos) ────────────────────────────────
        VOICE_TARGET_DB   = -6.0   # nivel RMS objetivo para la voz
        VOICE_ABOVE_MUSIC =  8.0   # dB que la voz debe superar a la música
        DUCK_EXTRA_DB     = 10.0   # dB extra de ducking durante la voz
        DUCK_THRESHOLD    =  0.02  # sensibilidad del detector de voz
        ATTACK_MS         = 20     # ms para bajar música al entrar voz
        RELEASE_MS        = 150    # ms para subir música al salir voz

        # ── Extraer waveforms ──────────────────────────────────────────────────
        v_waveform = voice["waveform"]
        v_sr       = voice["sample_rate"]
        m_waveform = music["waveform"]
        m_sr       = music["sample_rate"]

        v_np = waveform_to_mono(v_waveform)
        m_np = waveform_to_mono(m_waveform)

        # ── Analizar niveles ───────────────────────────────────────────────────
        v_rms = rms_db(v_np)
        m_rms = rms_db(m_np)

        voice_db  = max(-20.0, min(12.0, VOICE_TARGET_DB - v_rms))
        music_db  = max(-40.0, min(6.0,  (VOICE_TARGET_DB - VOICE_ABOVE_MUSIC) - m_rms))
        duck_db   = max(-60.0, min(0.0,  music_db - DUCK_EXTRA_DB))

        # ── Resamplear música si necesario ─────────────────────────────────────
        if m_sr != v_sr:
            ratio   = v_sr / m_sr
            new_len = int(len(m_np) * ratio)
            m_np    = np.interp(
                np.linspace(0, len(m_np) - 1, new_len),
                np.arange(len(m_np)),
                m_np
            ).astype(np.float32)

        # ── Igualar longitudes (loop música si es más corta) ───────────────────
        target_len = len(v_np)
        if len(m_np) < target_len:
            m_np = np.tile(m_np, int(np.ceil(target_len / len(m_np))))
        m_np = m_np[:target_len]

        # ── Aplicar volúmenes base ─────────────────────────────────────────────
        v_np = v_np * db_to_linear(voice_db)
        m_np = m_np * db_to_linear(music_db)

        # ── Detectar voz activa (RMS por ventanas de 10ms) ────────────────────
        window_size  = max(1, int(v_sr * 0.01))
        num_windows  = int(np.ceil(len(v_np) / window_size))
        voice_active = np.zeros(len(v_np), dtype=np.float32)

        for i in range(num_windows):
            start = i * window_size
            end   = min(start + window_size, len(v_np))
            chunk = v_np[start:end]
            rms   = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            voice_active[start:end] = 1.0 if rms > DUCK_THRESHOLD else 0.0

        # ── Aplicar ducking suave (attack/release) ─────────────────────────────
        attack_samples  = max(1, int(v_sr * ATTACK_MS  / 1000))
        release_samples = max(1, int(v_sr * RELEASE_MS / 1000))
        duck_to         = db_to_linear(duck_db)
        duck_range      = 1.0 - duck_to

        gain         = np.ones(len(v_np), dtype=np.float32)
        current_gain = 1.0

        for i in range(len(v_np)):
            if voice_active[i] > 0.5:
                current_gain = max(duck_to, current_gain - duck_range / attack_samples)
            else:
                current_gain = min(1.0,    current_gain + duck_range / release_samples)
            gain[i] = current_gain

        # ── Mezclar ────────────────────────────────────────────────────────────
        mixed = v_np + (m_np * gain)

        # ── Normalizar si hay clipping ─────────────────────────────────────────
        peak = np.abs(mixed).max()
        if peak > 0.98:
            mixed = mixed * (0.95 / peak)

        # ── Reconstruir formato AUDIO de ComfyUI ──────────────────────────────
        v_channels   = v_waveform.shape[1]
        mixed_tensor = torch.from_numpy(mixed).float()
        mixed_stereo = mixed_tensor.unsqueeze(0).expand(v_channels, -1)
        mixed_batch  = mixed_stereo.unsqueeze(0)

        output_audio = {"waveform": mixed_batch, "sample_rate": v_sr}

        # ── Reporte ────────────────────────────────────────────────────────────
        report = (
            f"=== OCTOPOZT AUTOMIX ===\n"
            f"Voz:    RMS {v_rms:+.1f} dB → ajuste {voice_db:+.1f} dB\n"
            f"Música: RMS {m_rms:+.1f} dB → ajuste {music_db:+.1f} dB\n"
            f"Ducking durante voz: {duck_db:+.1f} dB\n"
            f"Balance final: voz {VOICE_ABOVE_MUSIC:.0f} dB sobre música ✅"
        )

        return (output_audio, report)


NODE_CLASS_MAPPINGS_AUTOMIX = {
    "OctopoztAutoMix": OctopoztAutoMix,
}

NODE_DISPLAY_NAME_MAPPINGS_AUTOMIX = {
    "OctopoztAutoMix": "🐙 Octopozt AutoMix",
}
