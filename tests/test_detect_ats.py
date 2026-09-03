"""Fingerprint table: good URLs for all four ATSes plus near-misses that
must NOT match (spec acceptance: detection table incl. near-miss URLs)."""
import source_registry as sr

GOOD = [
    ("https://boards.greenhouse.io/stripe", ("greenhouse", "stripe")),
    ("https://boards.greenhouse.io/stripe/jobs/4247", ("greenhouse", "stripe")),
    # Detection is scheme-agnostic; the https rule lives in verify_source.
    ("http://boards.greenhouse.io/acmecorp", ("greenhouse", "acmecorp")),
    ("https://jobs.ashbyhq.com/eliseai", ("ashby", "eliseai")),
    ("https://jobs.ashbyhq.com/eliseai?utm_source=x", ("ashby", "eliseai")),
    ("https://JOBS.ASHBYHQ.COM/eliseai", ("ashby", "eliseai")),  # host case-insensitive
    ("https://jobs.lever.co/acme", ("lever", "acme")),
    ("https://jobs.lever.co/acme/", ("lever", "acme")),  # trailing slash
    ("https://jobs.smartrecruiters.com/Visa", ("smartrecruiters", "Visa")),  # token case preserved
    ("https://jobs.smartrecruiters.com/acme-corp", ("smartrecruiters", "acme-corp")),
]

NEAR_MISSES = [
    "https://boards.greenhouse.io",  # no token
    "https://boards.greenhouse.io/",  # no token
    "https://greenhouse.io/acme",  # missing boards. subdomain
    "https://jobs.ashbyhq.com",  # no token
    "https://ashbyhq.com/jobs/eliseai",  # right token, wrong host shape
    "https://evil.com/boards.greenhouse.io/acme",  # fingerprint smuggled into a foreign path
    "https://evil.com/?u=https://jobs.lever.co/acme",  # fingerprint smuggled into a query string
    "https://careers.acme.com/jobs",  # unknown ATS
    "https://boards.greenhouse.io:8080/acme",  # port in netloc — not a plain board URL
    "",
    "not a url at all",
]


def test_fingerprints_match():
    for url, expected in GOOD:
        assert sr.detect_ats(url) == expected, url


def test_near_misses_do_not_match():
    for url in NEAR_MISSES:
        assert sr.detect_ats(url) is None, url


def test_each_fingerprint_only_claims_its_own_host():
    for ats in ("greenhouse", "ashby", "lever", "smartrecruiters"):
        url = {
            "greenhouse": "https://boards.greenhouse.io/acme",
            "ashby": "https://jobs.ashbyhq.com/acme",
            "lever": "https://jobs.lever.co/acme",
            "smartrecruiters": "https://jobs.smartrecruiters.com/acme",
        }[ats]
        assert sr.detect_ats(url) == (ats, "acme")
