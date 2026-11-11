import os
from pathlib import Path

from mtg_recognition.settings import Settings


def test_rembg_finds_its_models_under_models_dir(monkeypatch):
    # Settings writes U2NET_HOME to os.environ; monkeypatch restores it afterwards
    monkeypatch.setenv("U2NET_HOME", "")

    Settings(MODELS_DIR=Path("elsewhere"))

    assert os.environ["U2NET_HOME"] == "elsewhere/segmentation/prod"


def test_thread_count_defaults_to_4(monkeypatch):
    monkeypatch.delenv("THREAD_COUNT", raising=False)

    assert Settings(_env_file=None).THREAD_COUNT == 4


def test_thread_count_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("THREAD_COUNT", "8")

    assert Settings(_env_file=None).THREAD_COUNT == 8


def test_openmp_uses_the_thread_count(monkeypatch):
    # Settings writes OMP_NUM_THREADS to os.environ; monkeypatch restores it
    monkeypatch.setenv("OMP_NUM_THREADS", "")

    Settings(_env_file=None, THREAD_COUNT=6)

    assert os.environ["OMP_NUM_THREADS"] == "6"
