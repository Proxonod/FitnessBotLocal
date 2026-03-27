from __future__ import annotations
"""
Voice: Lokale Spracherkennung via faster-whisper.

Strategie:
  1. small model (schnell, ~1GB VRAM)
  2. Falls avg_confidence < CONFIDENCE_THRESHOLD -> medium model (Fallback)
  3. User kann "nochmal" schreiben -> erzwingt medium

Gibt Text + Confidence zurueck.
"""

import io
import numpy as np
from faster_whisper import WhisperModel


CONFIDENCE_THRESHOLD = 0.75
SMALL_MODEL = "small"
MEDIUM_MODEL = "medium"

# Fuellwoerter die rausgefiltert werden
FILLER_WORDS = {
    "ähm", "äh", "uhm", "uh", "hmm", "hm", "also", "halt",
    "quasi", "sozusagen", "irgendwie", "eigentlich", "ja",
    "ne", "nee", "genau", "okay", "ok", "so", "und",
}


class VoiceRecognizer:
    def __init__(self):
        print("  [Voice] Loading Whisper small...")
        self._small = None
        self._medium = None

    def _get_small(self) -> WhisperModel:
        if self._small is None:
            self._small = WhisperModel(
                SMALL_MODEL,
                device="cuda",
                compute_type="float16",
            )
            print("  [Voice] Whisper small loaded")
        return self._small

    def _get_medium(self) -> WhisperModel:
        if self._medium is None:
            print("  [Voice] Loading Whisper medium (fallback)...")
            self._medium = WhisperModel(
                MEDIUM_MODEL,
                device="cuda",
                compute_type="float16",
            )
            print("  [Voice] Whisper medium loaded")
        return self._medium

    def _transcribe(self, model: WhisperModel,
                    audio_bytes: bytes) -> tuple[str, float]:
        """
        Transkribiert Audio-Bytes.
        Returns: (text, avg_confidence)
        """
        # faster-whisper braucht eine Datei oder numpy array
        audio_buf = io.BytesIO(audio_bytes)

        segments, info = model.transcribe(
            audio_buf,
            language="de",
            beam_size=5,
            vad_filter=True,          # Stille rausfiltern
            vad_parameters=dict(
                min_silence_duration_ms=500
            ),
        )

        # Segments sammeln
        all_text = []
        all_probs = []

        for segment in segments:
            all_text.append(segment.text.strip())
            # avg_logprob ist negativ, in 0-1 konvertieren
            confidence = min(1.0, max(0.0, np.exp(segment.avg_logprob)))
            all_probs.append(confidence)

        text = " ".join(all_text).strip()
        avg_confidence = float(np.mean(all_probs)) if all_probs else 0.0

        return text, avg_confidence

    def _filter_fillers(self, text: str) -> str:
        """Entfernt Fuellwoerter aus dem transkribierten Text."""
        words = text.split()
        filtered = [w for w in words
                    if w.lower().rstrip(".,!?") not in FILLER_WORDS]
        return " ".join(filtered)

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transkribiert Audio mit automatischem Fallback auf medium.
        Kein manueller Trigger noetig.
        """
        print("  [Voice] Transcribing with small...")
        raw_text, confidence = self._transcribe(self._get_small(), audio_bytes)
        model_used = "small"

        print(f"  [Voice] small confidence: {confidence:.2f}")

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  [Voice] Low confidence -> medium fallback")
            raw_text, confidence = self._transcribe(self._get_medium(), audio_bytes)
            model_used = "medium"
            print(f"  [Voice] medium confidence: {confidence:.2f}")

        filtered_text = self._filter_fillers(raw_text)
        print(f"  [Voice] Result ({model_used}, {confidence:.2f}): {filtered_text[:80]}")

        return {
            "text": filtered_text,
            "raw_text": raw_text,
            "confidence": confidence,
            "model_used": model_used,
        }