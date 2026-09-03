"""
Live verification probe — runs the real detect -> verify gate against real
boards and prints each raw HTTP status code + job count (the acceptance
evidence for the verification gate and the SmartRecruiters fetcher).

Usage:
  python live_probe.py                       # one Greenhouse + one Ashby board
  python live_probe.py <careers-url> [...]   # probe any boards you want

Each probe makes two live calls: one raw GET (to show the API's own status
code) and one through verify_source (the product's gate).
"""

import sys

import requests

import source_registry as sr

# Real, live boards — the default evidence targets for the PR report.
DEFAULT_PROBES = [
    "https://boards.greenhouse.io/stripe",  # real Greenhouse board
    "https://jobs.ashbyhq.com/eliseai",     # real Ashby board
]


def probe(url):
    detected = sr.detect_ats(url)
    if not detected:
        print(f"[detect] {url} -> no fingerprint match")
        return None
    ats, token = detected
    print(f"[detect] {url} -> ats={ats} token={token}")

    endpoint = sr.ENDPOINT_TEMPLATES[ats].format(token=token)
    try:
        resp = requests.get(endpoint, timeout=15)
        print(f"[live  ] GET {endpoint} -> HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"[live  ] GET {endpoint} -> network error: {e}")

    result = sr.verify_source({"endpoint": endpoint})
    if result.ok:
        print(f"[gate  ] PASS — {len(result.jobs)} job-like posting(s)")
    else:
        print(f"[gate  ] FAIL — reason: {result.reason}")
    return result


def main():
    urls = sys.argv[1:] or DEFAULT_PROBES
    print(f"Probing {len(urls)} board(s)...\n")
    results = [probe(url) for url in urls]
    print("\nSummary:")
    for url, result in zip(urls, results):
        if result is None:
            print(f"  {url}: not detected")
        elif result.ok:
            print(f"  {url}: PASS ({len(result.jobs)} jobs)")
        else:
            print(f"  {url}: FAIL ({result.reason})")


if __name__ == "__main__":
    main()
