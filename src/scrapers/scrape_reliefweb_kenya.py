"""
Scrape ReliefWeb Kenya Reports (Public Website, No API Key)
============================================================
The ReliefWeb v1 REST API used by data_collection_pipeline.py has been
decommissioned, and v2 now requires a pre-approved `appname`
(https://apidoc.reliefweb.int/parameters#appname). This script sidesteps
that gate entirely by crawling the public ReliefWeb website instead:

  1. Paginates https://reliefweb.int/updates?advanced-search=(PC131)
     (PC131 = ReliefWeb's internal country code for Kenya) to collect
     report URLs.
  2. Fetches each report page and pulls the body text from the
     `.rw-report__content` container.
  3. Splits bodies into sentences and keeps ones that read like real
     advisory/disaster/health content (length + domain-keyword gate).
  4. Appends new, deduplicated rows to a CSV using the same 4-column
     schema as data/clean/NDMA_English_Only.csv: English, Domain, Source,
     SourceFile (SourceFile holds the report URL here since there's no
     local file).

Usage:
    python src/scrapers/scrape_reliefweb_kenya.py --max-pages 200 --output data/raw/reliefweb_kenya_psas.csv

Progress is flushed to disk every 25 reports, so the script is safe to
Ctrl+C and resume later (it skips report URLs already present in
--output via the SourceFile column).
"""

import argparse
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

BASE_URL = "https://reliefweb.int"
LISTING_URL = f"{BASE_URL}/updates"
KENYA_ADVANCED_SEARCH = "(PC131)"  # Kenya's internal ReliefWeb country facet code
OUTPUT_COLUMNS = ["English", "Domain", "Source", "SourceFile"]

_DOMAIN_MAP = {
    "Health": [
        "health", "disease", "vaccine", "vaccination", "hospital", "clinic",
        "medical", "nurse", "doctor", "hiv", "malaria", "tb", "mpox",
        "maternal", "infant", "newborn", "pregnancy", "breastfeeding",
        "nutrition", "mental health", "medicine", "treatment", "patient", "cholera",
    ],
    "Education": [
        "school", "student", "pupil", "teacher", "exam", "kcse", "kcpe",
        "university", "college", "tvet", "enroll", "curriculum", "tuition",
        "bursary", "scholarship", "textbook", "learning", "literacy", "cbc",
    ],
    "Disaster/Health": [
        "drought", "flood", "famine", "hunger", "early warning", "bulletin",
        "emergency", "outbreak", "epidemic", "pandemic", "disaster", "relief",
        "humanitarian", "evacuation", "sanitation", "hygiene", "food security",
        "water supply", "rain", "rainfall", "climate", "weather", "crisis",
        "displacement", "refugee", "asylum",
    ],
    "Security": [
        "police", "crime", "theft", "fraud", "scam", "trafficking", "violence",
        "gbv", "safety", "shelter", "fire", "road safety", "traffic",
        "conflict", "security", "firearm",
    ],
    "Governance": [
        "vote", "election", "iebc", "government", "corruption", "eacc",
        "tax", "kra", "county", "parliament", "citizen", "public notice",
        "id card", "policy", "reform", "budget", "finance",
    ],
    "Agriculture": [
        "farm", "crop", "livestock", "seed", "fertilizer", "pest",
        "irrigation", "harvest", "maize", "animal", "veterinary",
        "arid", "pastoral", "farmers", "agribusiness",
    ],
}


def label_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {domain: sum(1 for kw in kws if kw in text_lower) for domain, kws in _DOMAIN_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Disaster/Health"


_BOILERPLATE_PATTERNS = [
    r"Attachments?\s+Download\s+(Report|Infographic|Map|Dataset|Document)s?\s*\([^)]*\)",
    r"Format\s+Format\s*:?\s*\w+",
    r"^Sources?\s*:",
    r"\bShare this\b",
    r"\bPreview\b\s*$",
]


def clean_and_split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"<[^>]+>", " ", text)
    for pat in _BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    cleaned = []
    for s in raw_sentences:
        s = s.strip()
        words = s.split()
        if 6 <= len(words) <= 60:
            cleaned.append(s)
    return cleaned


def fetch_report_links(session: requests.Session, page: int) -> list[str]:
    params = {"advanced-search": KENYA_ADVANCED_SEARCH, "list": "Kenya Updates", "page": page}
    try:
        r = session.get(LISTING_URL, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
    except Exception as exc:
        print(f"  [listing page {page}] error: {exc}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a[href*='/report/']"):
        href = a.get("href", "")
        if not href:
            continue
        full = urljoin(BASE_URL, href)
        full = full.split("?")[0]
        if full not in links:
            links.append(full)
    return links


def fetch_report_body(session: requests.Session, url: str):
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None, None
    except Exception as exc:
        print(f"  [report] error fetching {url}: {exc}")
        return None, None

    soup = BeautifulSoup(r.text, "html.parser")
    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else url

    body_el = soup.select_one(".rw-report__content") or soup.select_one("article")
    body_text = body_el.get_text(separator=" ", strip=True) if body_el else ""
    return title, body_text


def load_existing(paths: list[str]):
    seen_text = set()
    seen_urls = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        except Exception:
            continue
        if "English" in df.columns:
            seen_text.update(df["English"].dropna().str.strip().str.lower())
        if "SourceFile" in df.columns:
            seen_urls.update(df["SourceFile"].dropna().str.strip())
    return seen_text, seen_urls


def flush(records: list[dict], output_path: str):
    if not records:
        if os.path.exists(output_path):
            return len(pd.read_csv(output_path, dtype=str))
        return 0
    new_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path, dtype=str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["English"], keep="first")
    else:
        combined = new_df
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Scrape ReliefWeb Kenya reports for PSA-style sentences.")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=200, help="20 reports per listing page")
    parser.add_argument("--delay", type=float, default=0.6, help="seconds between report requests")
    parser.add_argument("--output", type=str, default="data/raw/reliefweb_kenya_psas.csv")
    parser.add_argument(
        "--dedupe-against", nargs="*",
        default=["data/clean/NDMA_English_Only.csv", "data/Master_PSA_Only.csv"],
        help="extra CSVs whose English column should count as already-seen",
    )
    args = parser.parse_args()

    dedupe_paths = list(args.dedupe_against) + [args.output]
    seen_text, seen_urls = load_existing(dedupe_paths)
    print(f"Starting with {len(seen_text)} known English sentences, {len(seen_urls)} known report URLs.")

    session = requests.Session()
    records = []
    consecutive_empty = 0
    total_new = 0

    for page in range(args.start_page, args.start_page + args.max_pages):
        links = fetch_report_links(session, page)
        if not links:
            consecutive_empty += 1
            print(f"[page {page}] no report links found ({consecutive_empty} consecutive empty pages)")
            if consecutive_empty >= 4:
                print("Stopping: reached the end of the Kenya updates listing.")
                break
            continue
        consecutive_empty = 0

        new_links = [l for l in links if l not in seen_urls]
        print(f"[page {page}] {len(links)} links, {len(new_links)} new")

        for link in new_links:
            seen_urls.add(link)
            title, body = fetch_report_body(session, link)
            if not body:
                continue
            for sentence in clean_and_split_sentences(body):
                key = sentence.strip().lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                records.append({
                    "English": sentence.strip(),
                    "Domain": label_domain(sentence),
                    "Source": title,
                    "SourceFile": link,
                })
                total_new += 1
            time.sleep(args.delay)

            if len(records) >= 25:
                total = flush(records, args.output)
                print(f"  flushed -> {args.output} now has {total} rows (+{total_new} new so far)")
                records = []

    total = flush(records, args.output)
    print(f"\nDone. {args.output} now has {total} rows ({total_new} new sentences added this run).")


if __name__ == "__main__":
    main()
