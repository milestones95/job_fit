"""SmartRecruiters fetcher: field mapping + offset/limit pagination."""
import job_fit_finder as jf


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def posting(idx):
    return {
        "id": f"00000{idx}",
        "name": f"Engineer {idx}",
        "location": {"city": "New York", "region": "NY", "country": "us"},
        "department": {"label": "Engineering"},
        "company": {"name": "Acme"},
    }


def patch(monkeypatch, pages):
    """pages: list of response bodies served in request order. Records calls."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(jf.requests, "get", fake_get)
    return calls


def test_single_page_mapping(monkeypatch):
    calls = patch(monkeypatch, [{
        "content": [posting(1), posting(2)],
        "offset": 0, "limit": 100, "totalFound": 2,
    }])

    jobs = jf.fetch_smartrecruiters("Acme", "acme")

    assert len(calls) == 1
    assert calls[0][0] == "https://api.smartrecruiters.com/v1/companies/acme/postings"
    assert len(jobs) == 2
    job = jobs[0]
    assert job["title"] == "Engineer 1"            # name -> title
    assert job["department"] == "Engineering"      # department.label -> department
    assert job["location"] == "New York, NY, us"
    assert job["url"] == "https://jobs.smartrecruiters.com/acme/000001"
    assert job["job_id"] == "000001"
    assert job["company"] == "Acme"
    assert job["description"] == ""                # list endpoint has no description
    assert job["compensation"] == ""


def test_pagination_walks_pages_until_total_found(monkeypatch):
    calls = patch(monkeypatch, [
        {"content": [posting(1), posting(2)], "offset": 0, "limit": 2, "totalFound": 3},
        {"content": [posting(3)], "offset": 2, "limit": 2, "totalFound": 3},
    ])

    jobs = jf.fetch_smartrecruiters("Acme", "acme")

    assert len(calls) == 2
    assert calls[0][1] == {"offset": 0, "limit": 100}
    assert calls[1][1] == {"offset": 2, "limit": 100}  # offset advanced by page size
    assert [j["title"] for j in jobs] == ["Engineer 1", "Engineer 2", "Engineer 3"]


def test_pagination_stops_on_short_page_without_total_found(monkeypatch):
    calls = patch(monkeypatch, [
        {"content": [posting(1)]},  # fewer than limit -> last page
    ])

    jobs = jf.fetch_smartrecruiters("Acme", "acme")

    assert len(calls) == 1
    assert len(jobs) == 1


def test_empty_board_returns_empty_list(monkeypatch):
    patch(monkeypatch, [{"content": [], "offset": 0, "limit": 100, "totalFound": 0}])
    assert jf.fetch_smartrecruiters("Acme", "acme") == []
