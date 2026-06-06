from voxlocal.stt._sensevoice import SenseVoiceSTT


def test_sensevoice_stt_has_transcribe():
    stt = SenseVoiceSTT.__new__(SenseVoiceSTT)
    assert hasattr(stt, "transcribe")


def test_sensevoice_stt_language():
    stt = SenseVoiceSTT.__new__(SenseVoiceSTT)
    stt.language = "ja"
    assert stt.language == "ja"
