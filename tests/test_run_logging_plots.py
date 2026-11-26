import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ml import run_logging
from ml.run_logging import EpochRecorder, plot_confusion, plot_curves

ROOT = Path(__file__).resolve().parent.parent
needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib tidak terpasang"
)


@needs_matplotlib
def test_plots_work_even_when_mplbackend_is_the_notebook_inline_backend(tmp_path: Path):
    """Regresi dari Colab: MPLBACKEND=module://matplotlib_inline... membuat `import matplotlib` gagal."""
    history = tmp_path / "history.csv"
    recorder = EpochRecorder(history, "defect", "head")
    recorder.record(1, {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.4, "val_accuracy": 0.85}, 1e-3, 1.0)
    recorder.record(2, {"loss": 0.4, "accuracy": 0.9, "val_loss": 0.35, "val_accuracy": 0.9}, 1e-3, 1.0)
    script = (
        "from pathlib import Path\n"
        "from ml.run_logging import plot_curves, plot_confusion\n"
        f"out = Path(r'{tmp_path}')\n"
        f"print(len(plot_curves(Path(r'{history}'), out)))\n"
        "m = {'classes': ['a', 'b'], 'confusion_matrix': [[3, 1], [0, 4]]}\n"
        "print(plot_confusion(m, out / 'cm.png', 't') is not None)\n"
    )
    env = {**__import__("os").environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline",
           "PYTHONPATH": str(ROOT)}

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=ROOT)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "True"]
    assert (tmp_path / "curves_defect.png").is_file() and (tmp_path / "cm.png").is_file()


def test_a_plotting_error_never_aborts_training(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(run_logging, "_pyplot", lambda: object())

    def boom(*args, **kwargs):
        raise RuntimeError("gagal menggambar")

    monkeypatch.setattr(run_logging, "_draw_curves", boom)
    monkeypatch.setattr(run_logging, "_draw_confusion", boom)

    with caplog.at_level(logging.WARNING, logger="ml.run_logging"):
        assert plot_curves(tmp_path / "h.csv", tmp_path) == []
        assert plot_confusion({"classes": [], "confusion_matrix": []}, tmp_path / "c.png", "t") is None

    assert "dilewati" in caplog.text
