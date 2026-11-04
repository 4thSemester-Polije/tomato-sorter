import ntpath
from types import SimpleNamespace

import pytest

from ml.resnext import posix_keras_paths


def _fake_tf(join, share_module=False):
    public = SimpleNamespace(join=join)
    compat = public if share_module else SimpleNamespace(join=join)
    tf = SimpleNamespace(
        io=SimpleNamespace(gfile=public),
        compat=SimpleNamespace(v2=SimpleNamespace(io=SimpleNamespace(gfile=compat))),
    )
    return tf, public, compat


def test_relative_keras_paths_use_forward_slash_and_are_restored():
    tf, public, compat = _fake_tf(ntpath.join)

    with posix_keras_paths(tf, enabled=True):
        assert public.join("layers", "resnext50") == "layers/resnext50"
        assert compat.join("layers/resnext50", "layers") == "layers/resnext50/layers"
        assert compat.join("", "layers") == "layers"

    assert public.join("layers", "x") == "layers\\x"
    assert compat.join("layers", "x") == "layers\\x"


def test_absolute_paths_and_urls_keep_native_join():
    calls = []

    def native(path, *paths):
        calls.append(path)
        return "native"

    tf, public, _ = _fake_tf(native)

    with posix_keras_paths(tf, enabled=True):
        assert public.join("/tmp/keras", "assets") == "native"
        assert public.join("gs://bucket", "file") == "native"

    assert calls == ["/tmp/keras", "gs://bucket"]


def test_disabled_leaves_join_untouched():
    tf, public, compat = _fake_tf(ntpath.join)
    original = public.join

    with posix_keras_paths(tf, enabled=False):
        assert public.join is original
        assert compat.join is original


def test_join_is_restored_even_when_loading_fails():
    tf, public, _ = _fake_tf(ntpath.join)
    original = public.join

    with pytest.raises(RuntimeError):
        with posix_keras_paths(tf, enabled=True):
            raise RuntimeError("load failed")

    assert public.join is original


def test_shared_gfile_module_is_patched_and_restored_once():
    tf, public, compat = _fake_tf(ntpath.join, share_module=True)
    assert public is compat

    with posix_keras_paths(tf, enabled=True):
        assert public.join("a", "b") == "a/b"

    assert public.join("a", "b") == "a\\b"
