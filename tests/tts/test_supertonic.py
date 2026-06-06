from voxlocal.tts._supertonic import SupertonicTTS


def test_supertonic_tts_has_speak():
    tts = SupertonicTTS.__new__(SupertonicTTS)
    assert hasattr(tts, "speak")


def test_supertonic_tts_has_speak_iter():
    tts = SupertonicTTS.__new__(SupertonicTTS)
    assert hasattr(tts, "speak_iter")


def test_supertonic_tts_language():
    tts = SupertonicTTS.__new__(SupertonicTTS)
    tts.language = "es"
    assert tts.language == "es"
