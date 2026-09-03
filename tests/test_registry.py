"""Registry round-trip: versioned sources.json, built-in seeding, atomic
writes, idempotent upserts, built-ins-first/registry-overrides merging."""
import json

import pytest

import source_registry as sr


NATIVE_KEYS = {"id", "company", "ats", "board_token", "endpoint", "method",
               "headers", "added_via", "added_at", "verification"}


def test_seeds_builtins_on_first_run(registry_path):
    assert not registry_path.exists()
    data = sr.load_registry()
    assert registry_path.exists()
    assert data["version"] == 1
    assert [s["id"] for s in data["sources"]] == [
        "eliseai", "browserbase", "tamarindbio", "decagon",
    ]
    for entry in data["sources"]:
        assert set(entry.keys()) == NATIVE_KEYS
        assert entry["added_via"] == "builtin"
        assert entry["verification"]["status"] == "builtin"
        assert entry["endpoint"].startswith("https://")  # built from ENDPOINT_TEMPLATES


def test_round_trip_upsert_then_load(registry_path):
    entry = {
        "id": "acme", "company": "Acme", "ats": "lever", "board_token": "acme",
        "endpoint": "https://api.lever.co/v0/postings/acme?mode=json",
        "method": "GET", "headers": {}, "added_via": "extension",
        "added_at": "2026-09-03T00:00:00Z",
        "verification": {"status": "passed", "job_count": 3, "checked_at": "2026-09-03T00:00:00Z"},
    }
    saved = sr.upsert_source(entry)
    loaded = sr.load_registry()
    assert loaded["sources"][-1] == saved
    assert saved == entry  # round-trip fidelity


def test_upsert_dedupes_on_ats_and_token(registry_path):
    sr.load_registry()  # seed
    updated = sr.upsert_source({
        "id": "eliseai", "company": "Ignored", "ats": "ashby", "board_token": "eliseai",
        "endpoint": "https://api.ashbyhq.com/posting-api/job-board/eliseai",
        "method": "GET", "headers": {},
        "verification": {"status": "passed", "job_count": 9, "checked_at": "2026-09-03T01:00:00Z"},
    })
    data = sr.load_registry()
    assert len(data["sources"]) == 4  # no duplicate
    assert data["sources"][0]["verification"]["job_count"] == 9
    # original identity preserved on refresh
    assert updated["company"] == "EliseAI"
    assert updated["added_via"] == "builtin"


def test_id_collision_gets_ats_suffix(registry_path):
    sr.load_registry()  # seeds id "eliseai" (ashby)
    sr.upsert_source({
        "id": "eliseai", "company": "Eliseai", "ats": "greenhouse", "board_token": "eliseai",
        "endpoint": "https://boards-api.greenhouse.io/v1/boards/eliseai/jobs?content=true",
        "method": "GET", "headers": {},
        "verification": {"status": "passed", "job_count": 1, "checked_at": "2026-09-03T01:00:00Z"},
    })
    data = sr.load_registry()
    ids = [s["id"] for s in data["sources"]]
    assert len(ids) == len(set(ids))  # ids unique
    assert "eliseai-greenhouse" in ids


def test_corrupt_registry_raises_instead_of_overwriting(registry_path):
    registry_path.write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        sr.load_registry()
    assert registry_path.read_text() == "{not json"  # untouched


def test_wrong_shape_registry_raises(registry_path):
    registry_path.write_text(json.dumps({"sources": "not-a-list"}))
    with pytest.raises(RuntimeError, match="unexpected shape"):
        sr.load_registry()


def test_empty_file_is_seeded(registry_path):
    registry_path.write_text("")
    data = sr.load_registry()
    assert len(data["sources"]) == 4


def test_atomic_write_leaves_no_tmp_residue(registry_path):
    sr.load_registry()
    sr.upsert_source({
        "id": "acme", "company": "Acme", "ats": "lever", "board_token": "acme",
        "endpoint": "x", "method": "GET", "headers": {},
        "verification": {"status": "passed", "job_count": 1, "checked_at": "x"},
    })
    assert list(registry_path.parent.glob(".sources-*")) == []
    json.loads(registry_path.read_text())  # still valid JSON


def test_get_sources_builtins_first_registry_overrides(registry_path):
    import job_fit_finder as jf

    jf.get_sources()  # seeds via load_registry
    # Override a built-in in the registry file: same id, different board token.
    data = sr.load_registry()
    data["sources"][0]["board_token"] = "eliseai-v2"
    sr.save_registry(data)

    merged = jf.get_sources()
    assert merged["eliseai"]["token"] == "eliseai-v2"  # registry wins
    assert merged["decagon"]["token"] == "decagon"     # built-ins still present


def test_get_sources_skips_snippet_entries(registry_path, capsys):
    import job_fit_finder as jf

    jf.get_sources()
    data = sr.load_registry()
    data["sources"].append({
        "id": "acme-custom", "company": "Acme Custom",
        "adapter": {"kind": "snippet", "language": "python", "code": "def fetch_jobs():\n    return []"},
        "added_via": "extension", "added_at": "2026-09-03T00:00:00Z",
        "verification": {"status": "passed", "job_count": 1, "checked_at": "2026-09-03T00:00:00Z"},
    })
    sr.save_registry(data)

    merged = jf.get_sources()
    assert "acme-custom" not in merged  # not dispatchable until the runner exists
    assert "snippet source" in capsys.readouterr().out
