"""
Octopozt Audio Mixer — ComfyUI Custom Node
Mezcla voz + música con ducking automático.
La música baja de volumen cuando detecta voz activa.
"""

import torch
import numpy as np


class OctopoztAudioMixer:
    """
    Mezcla dos pistas de audio (voz + música) con ducking automático.
    Cuando hay voz activa, la música baja automáticamente.
    """

    CATEGORY = "Octopozt"
    FUNCTION = "mix"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice":          ("AUDIO",),
                "music":          ("AUDIO",),
                "voice_volume":   ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 2.0, "step": 0.05}),
                "music_volume":   ("FLOAT", {"default": 0.4,  "min": 0.0, "max": 2.0, "step": 0.05}),
                "duck_to":        ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Volumen de música cuando hay voz (0.08 = 8%)"}),
                "duck_threshold": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001,
                                             "tooltip": "Sensibilidad de detección de voz"}),
                "attack_ms":      ("INT",   {"default": 20,   "min": 1,   "max": 200,
                                             "tooltip": "ms para bajar la música cuando entra la voz"}),
                "release_ms":     ("INT",   {"default": 150,  "min": 10,  "max": 1000,
                                             "tooltip": "ms para subir la música cuando termina la voz"}),
            }
        }

    def mix(self, voice, music, voice_volume, music_volume,
            duck_to, duck_threshold, attack_ms, release_ms):

        # ── Extraer waveforms ──────────────────────────────────────────────────
        v_waveform = voice["waveform"]   # (batch, channels, samples)
        v_sr       = voice["sample_rate"]
        m_waveform = music["waveform"]
        m_sr       = music["sample_rate"]

        # Convertir a numpy mono para procesar
        v_np = v_waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)
        m_np = m_waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)

        # ── Resamplear música si tiene diferente sample rate ───────────────────
        if m_sr != v_sr:
            ratio = v_sr / m_sr
            new_len = int(len(m_np) * ratio)
            m_np = np.interp(
                np.linspace(0, len(m_np) - 1, new_len),
                np.arange(len(m_np)),
                m_np
            ).astype(np.float32)

        # ── Igualar longitudes ─────────────────────────────────────────────────
        target_len = len(v_np)
        if len(m_np) < target_len:
            # Loop música si es más corta
            repeats = int(np.ceil(target_len / len(m_np)))
            m_np = np.tile(m_np, repeats)
        m_np = m_np[:target_len]

        # ── Aplicar volúmenes base ─────────────────────────────────────────────
        v_np = v_np * voice_volume
        m_np = m_np * music_volume

        # ── Calcular envelope de voz (RMS por ventana) ─────────────────────────
        window_size = max(1, int(v_sr * 0.01))  # ventanas de 10ms
        num_windows = int(np.ceil(len(v_np) / window_size))

        voice_active = np.zeros(len(v_np), dtype=np.float32)

        for i in range(num_windows):
            start = i * window_size
            end   = min(start + window_size, len(v_np))
            chunk = v_np[start:end]
            rms   = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            voice_active[start:end] = 1.0 if rms > duck_threshold else 0.0

        # ── Suavizar la señal de control (attack/release) ──────────────────────
        attack_samples  = max(1, int(v_sr * attack_ms  / 1000))
        release_samples = max(1, int(v_sr * release_ms / 1000))

        gain = np.ones(len(v_np), dtype=np.float32)
        current_gain = 1.0
        duck_range   = 1.0 - duck_to

        for i in range(len(v_np)):
            if voice_active[i] > 0.5:
                # Voz activa → bajar música (attack)
                step = duck_range / attack_samples
                current_gain = max(duck_to, current_gain - step)
            else:
                # Sin voz → subir música (release)
                step = duck_range / release_samples
                current_gain = min(1.0, current_gain + step)
            gain[i] = current_gain

        # ── Aplicar ducking a la música ────────────────────────────────────────
        m_ducked = m_np * gain

        # ── Mezclar ────────────────────────────────────────────────────────────
        mixed = v_np + m_ducked

        # ── Normalizar si hay clipping ─────────────────────────────────────────
        peak = np.abs(mixed).max()
        if peak > 0.98:
            mixed = mixed * (0.95 / peak)

        # ── Reconstruir formato AUDIO de ComfyUI ──────────────────────────────
        # Determinar canales del output (usar los de la voz)
        v_channels = v_waveform.shape[1]
        mixed_tensor = torch.from_numpy(mixed).float()
        mixed_stereo = mixed_tensor.unsqueeze(0).expand(v_channels, -1)
        mixed_batch  = mixed_stereo.unsqueeze(0)  # (1, channels, samples)

        output_audio = {
            "waveform":    mixed_batch,
            "sample_rate": v_sr,
        }

        return (output_audio,)


NODE_CLASS_MAPPINGS_MIXER = {
    "OctopoztAudioMixer": OctopoztAudioMixer,
}

NODE_DISPLAY_NAME_MAPPINGS_MIXER = {
    "OctopoztAudioMixer": "🐙 Octopozt Audio Mixer",
}
