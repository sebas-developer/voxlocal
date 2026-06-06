from voxlocal.stt._moonshine import MoonshineSTT


def test_moonshine_stt_has_transcribe():
    stt = MoonshineSTT.__new__(MoonshineSTT)
    assert hasattr(stt, "transcribe")


def test_moonshine_stt_language():
    stt = MoonshineSTT.__new__(MoonshineSTT)
    stt.language = "es"
    assert stt.language == "es"
