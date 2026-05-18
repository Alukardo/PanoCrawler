"""Guard: fail any test that mutates the production info.csv.

Tests must monkeypatch ``main.infoPath`` (and ``main.write_info_records`` when
faking ``download_missing_panorama``) so persistence touches a tmp path. If a
test escapes that contract and writes to the real NTUST CSV, this hook raises
with the offending nodeid so the regression is impossible to miss.

The guard is a no-op when the production CSV does not exist (e.g. CI / fresh
clones), so it does not impose any environmental requirement.
"""
import hashlib
import pathlib
import pytest

import main as _main


def _hash():
    path = _main.infoPath
    try:
        return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    before = _hash()
    yield
    after = _hash()
    if before is not None and before != after:
        raise AssertionError(
            f"Test {item.nodeid} mutated the production CSV at {_main.infoPath}. "
            "Add monkeypatch.setattr(main, 'infoPath', ...) and "
            "monkeypatch.setattr(main, 'write_info_records', lambda r: None) "
            "(or supply a real tmp file) to keep the test isolated."
        )
