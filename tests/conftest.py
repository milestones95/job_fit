import pytest


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    """Isolated sources.json for the test + a .env location that does not
    exist, so nothing reads the developer's real secrets or registry."""
    path = tmp_path / "sources.json"
    monkeypatch.setenv("JOB_FIT_SOURCES_PATH", str(path))
    monkeypatch.setenv("JOB_FIT_ENV_PATH", str(tmp_path / "no.env"))
    return path


@pytest.fixture
def seeded_registry(registry_path):
    """sources.json seeded from the built-ins (as it would be on first run)."""
    import source_registry as sr

    sr.load_registry()
    return registry_path
