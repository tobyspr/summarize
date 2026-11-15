"""Cloud Whisper reads the Groq key from GROQ_API_KEY or the legacy lower-case name."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from summarizer.transcription import _transcribe_cloud_whisper


@pytest.mark.parametrize("env_name", ["GROQ_API_KEY", "groq"])
def test_cloud_whisper_accepts_either_groq_env_name(env_name, monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("groq", raising=False)
    monkeypatch.setenv(env_name, "secret")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    fake_groq = types.ModuleType("groq")
    fake_groq.Groq = MagicMock()
    fake_groq.Groq.return_value.audio.transcriptions.create.return_value = "hola"
    with patch.dict(sys.modules, {"groq": fake_groq}):
        _transcribe_cloud_whisper(str(audio), verbose=False)

    fake_groq.Groq.assert_called_once_with(api_key="secret")
