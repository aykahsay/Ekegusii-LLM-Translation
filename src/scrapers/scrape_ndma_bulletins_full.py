"""
NDMA KnowledgeWeb Bulk Bulletin Scraper (run this LOCALLY, not from a sandboxed
CI/agent environment -- see note below)
=============================================================================
Downloads and extracts EVERY Drought Early Warning (DEW) Bulletin, Long/Short
Rains Assessment, and related advisory PDF published on the Kenya National
Drought Management Authority's document library:

    https://knowledgeweb.ndma.go.ke/

...for all ~23 ASAL counties and all available months/years -- the same
source the existing data/clean/NDMA_English_Only.csv sample (1,429 rows,
11 PDFs) was built from.

IMPORTANT NETWORK NOTE
-----------------------
`knowledgeweb.ndma.go.ke` was completely unreachable (connection refused/
timeout on port 80 and 443) from the sandboxed environment this script was
written in, even though the main `ndma.go.ke` site worked fine. This script
could therefore NOT be tested end-to-end against the live site. Run it on
your own machine/network -- if it was reachable when the 11 sample PDFs were
originally collected, it should be reachable for you too.

If a step below doesn't find anything, re-run with `--debug-dump` and send
back the saved HTML files in `data/raw/ndma_debug/` -- the selectors in
`find_pdf_links()` / postback handling are the most likely thing to need a
small adjustment for the site's actual current markup.

WHAT IT DOES
------------
1. DISCOVER  -- crawls known category listing pages
                (Public/Resources/Default.aspx?ID=1..60) plus a same-domain
                breadth-first crawl from the homepage, following ASP.NET
                GridView postback pagination where present, collecting every
                PDF link and every Library/doclink.aspx?document=<guid> link
                (resolving the latter to its underlying PDF).
                Manifest saved to data/raw/ndma_bulletin_manifest.csv so the
                run is resumable and inspectable.
2. DOWNLOAD  -- fetches each new PDF into data/raw/pdfs/ndma_dew_bulletins/,
                skipping files already on disk.
3. EXTRACT   -- pulls sentences out of each PDF (declarative bulletin prose,
                6-50 words, matching the style of the existing sample),
                labels Source/SourceFile/Domain the same way the existing
                1,429-row sample does, and appends new unique rows (deduped
                case-insensitively against the existing file) to
                data/clean/NDMA_English_Only.csv.

Usage:
    python src/scrapers/scrape_ndma_bulletins_full.py
    python src/scrapers/scrape_ndma_bulletins_full.py --skip-download   # re-extract only
    python src/scrapers/scrape_ndma_bulletins_full.py --debug-dump      # save raw HTML for troubleshooting
"""

import argparse
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pypdf import PdfReader

BASE_URL = "https://knowledgeweb.ndma.go.ke"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PDF_DIR = os.path.join("data", "raw", "pdfs", "ndma_dew_bulletins")
MANIFEST_PATH = os.path.join("data", "raw", "ndma_bulletin_manifest.csv")
OUTPUT_CSV = os.path.join("data", "clean", "NDMA_English_Only.csv")
# build_master_datasets.py only scans data/, data/raw/, data/intermediate/ (non-recursive),
# so it never sees data/clean/*.csv -- also drop the new rows here so a later
# `python src/data_processing/build_master_datasets.py` run picks them up automatically.
NEW_ROWS_FOR_MASTER_BUILD = os.path.join("data", "raw", "ndma_bulletins_new.csv")
DEBUG_DIR = os.path.join("data", "raw", "ndma_debug")
OUTPUT_COLUMNS = ["English", "Domain", "Source", "SourceFile"]

# Category IDs are a guess based on two confirmed examples found via search
# (ID=2 -> Annual Reports, ID=17 -> County Long Rain Assessments). Brute-force
# a generous range; empty/404 categories are skipped automatically.
CATEGORY_ID_RANGE = range(1, 61)

# Relevance keywords -- only download PDFs whose link text/filename suggests
# drought/disaster/food-security content (keeps the corpus on-topic and skips
# unrelated tenders/investment docs also hosted on the same portal).
RELEVANCE_KEYWORDS = [
    "dew", "drought", "early warning", "ews", "bulletin", "long rain",
    "short rain", "lra", "sra", "food security", "nutrition", "famine",
    "flood", "national drought", "livelihood", "phase classification",
]


def is_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in RELEVANCE_KEYWORDS)


# ---------------------------------------------------------------------------
# Step 1: Discovery (handles both plain links and ASP.NET postback pagers)
# ---------------------------------------------------------------------------

def get_hidden_fields(soup: BeautifulSoup) -> dict:
    fields = {}
    for inp in soup.select("input[type=hidden]"):
        name = inp.get("name")
        if name:
            fields[name] = inp.get("value", "")
    return fields


def find_postback_pages(soup: BeautifulSoup):
    """Return [(event_target, event_argument), ...] for GridView-style pagers."""
    targets = []
    for a in soup.select("a[href*='__doPostBack']"):
        m = re.search(r"__doPostBack\('([^']+)','([^']*)'\)", a.get("href", ""))
        if m:
            targets.append((m.group(1), m.group(2)))
    return targets


def find_pdf_links(soup: BeautifulSoup, page_url: str):
    """Collect (absolute_url, link_text) for direct PDFs and doclink wrappers."""
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True) or href
        if href.lower().endswith(".pdf"):
            found.append((urljoin(page_url, href), text))
        elif "doclink.aspx" in href.lower() or "resourcedetails.aspx" in href.lower():
            found.append((urljoin(page_url, href), text))
    return found


def resolve_doclink(session: requests.Session, url: str):
    """doclink.aspx / ResourceDetails.aspx wrap the actual PDF -- follow through."""
    try:
        r = session.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
    except Exception as exc:
        print(f"    doclink error {url}: {exc}")
        return None
    ctype = r.headers.get("Content-Type", "")
    if "pdf" in ctype.lower():
        return r.url
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            return urljoin(r.url, a["href"])
    for tag in soup.find_all(["iframe", "embed", "object"]):
        src = tag.get("src") or tag.get("data")
        if src and src.lower().endswith(".pdf"):
            return urljoin(r.url, src)
    return None


def crawl_listing(session: requests.Session, start_url: str, debug_dump: bool, max_postback_pages: int = 100):
    """Fetch a listing page and, if it's a postback-paginated GridView, page through it."""
    all_links = []
    try:
        r = session.get(start_url, headers=HEADERS, timeout=25)
    except Exception as exc:
        print(f"  error fetching {start_url}: {exc}")
        return all_links
    if r.status_code != 200:
        return all_links

    soup = BeautifulSoup(r.text, "html.parser")
    if debug_dump:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        fname = re.sub(r"[^\w]+", "_", start_url)[-120:] + ".html"
        with open(os.path.join(DEBUG_DIR, fname), "w", encoding="utf-8") as f:
            f.write(r.text)

    all_links.extend(find_pdf_links(soup, start_url))

    seen_args = set()
    for target, arg in find_postback_pages(soup):
        if arg in seen_args or len(seen_args) >= max_postback_pages:
            continue
        seen_args.add(arg)
        payload = get_hidden_fields(soup)
        payload["__EVENTTARGET"] = target
        payload["__EVENTARGUMENT"] = arg
        try:
            r2 = session.post(start_url, data=payload, headers=HEADERS, timeout=25)
        except Exception as exc:
            print(f"    postback error ({target},{arg}): {exc}")
            continue
        if r2.status_code != 200:
            continue
        soup2 = BeautifulSoup(r2.text, "html.parser")
        new_links = find_pdf_links(soup2, start_url)
        if not new_links:
            continue
        all_links.extend(new_links)
        # Newer postback pages may reveal further page targets -- pick up any new ones.
        for t2, a2 in find_postback_pages(soup2):
            if a2 not in seen_args:
                seen_args.add(a2)
        time.sleep(0.3)

    return all_links


def discover(session: requests.Session, debug_dump: bool) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: DISCOVER -- crawling knowledgeweb.ndma.go.ke for bulletin PDFs")
    print("=" * 70)

    existing_manifest = pd.DataFrame(columns=["url", "title"])
    if os.path.exists(MANIFEST_PATH):
        existing_manifest = pd.read_csv(MANIFEST_PATH, dtype=str)

    seen_urls = set(existing_manifest["url"]) if not existing_manifest.empty else set()
    rows = list(existing_manifest.to_dict("records"))

    entry_points = [BASE_URL + "/"] + [
        f"{BASE_URL}/Public/Resources/Default.aspx?ID={i}" for i in CATEGORY_ID_RANGE
    ]

    for url in entry_points:
        links = crawl_listing(session, url, debug_dump)
        if not links:
            continue
        relevant = [(u, t) for u, t in links if is_relevant(t) or is_relevant(u)]
        print(f"  {url} -> {len(links)} links total, {len(relevant)} relevant")
        for link_url, text in relevant:
            resolved = link_url
            if "doclink.aspx" in link_url.lower() or "resourcedetails.aspx" in link_url.lower():
                resolved = resolve_doclink(session, link_url)
                time.sleep(0.2)
            if not resolved or resolved in seen_urls:
                continue
            seen_urls.add(resolved)
            rows.append({"url": resolved, "title": text})
        time.sleep(0.4)

    manifest = pd.DataFrame(rows, columns=["url", "title"]).drop_duplicates(subset=["url"])
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"\nManifest saved: {len(manifest)} unique PDF URLs -> {MANIFEST_PATH}")
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Download
# ---------------------------------------------------------------------------

def download_all(session: requests.Session, manifest: pd.DataFrame, delay: float, max_downloads: int):
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOAD")
    print("=" * 70)
    os.makedirs(PDF_DIR, exist_ok=True)
    existing = set(os.listdir(PDF_DIR))
    downloaded = 0

    for _, row in manifest.iterrows():
        if downloaded >= max_downloads:
            print(f"  reached --max-downloads ({max_downloads}); stopping this run.")
            break
        url = row["url"]
        fname = os.path.basename(urlparse(url).path)
        if not fname.lower().endswith(".pdf"):
            fname = re.sub(r"[^\w.-]+", "_", row.get("title", "doc")) + ".pdf"
        if fname in existing:
            continue

        for attempt in range(3):
            try:
                r = session.get(url, headers=HEADERS, timeout=40)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    with open(os.path.join(PDF_DIR, fname), "wb") as f:
                        f.write(r.content)
                    existing.add(fname)
                    downloaded += 1
                    print(f"  [{downloaded}] saved {fname} ({len(r.content)//1024} KB)")
                    break
                else:
                    print(f"  attempt {attempt+1}: unexpected response for {url} (status {r.status_code})")
            except Exception as exc:
                print(f"  attempt {attempt+1} error downloading {url}: {exc}")
            time.sleep(1.5 * (attempt + 1))
        time.sleep(delay)

    print(f"\nDownloaded {downloaded} new PDFs. Folder now has {len(os.listdir(PDF_DIR))} files total.")


# ---------------------------------------------------------------------------
# Step 3: Extract -- mirrors the schema/derivation logic used to build the
# existing 1,429-row data/clean/NDMA_English_Only.csv sample.
# ---------------------------------------------------------------------------

def derive_source(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    # Strip the trailing CMS timestamp digits, e.g. "...April_202520250516125942"
    name = re.sub(r"\d{14}$", "", name)
    if "_DEW_Bulletin_" in name:
        county, _, rest = name.partition("_DEW_Bulletin_")
        county = county.replace("_", " ").strip()
        rest = rest.replace("_", " ").strip()
        return f"NDMA {county} Drought Early Warning Bulletin {rest}".strip()
    if "Long_Rains_Assessment" in name or "Long_Rain_Assessment" in name:
        county = name.split("_Long_Rain")[0].replace("_", " ").strip()
        return f"NDMA {county} Long Rains Assessment Report".strip()
    if "Short_Rains_Assessment" in name or "Short_Rain_Assessment" in name:
        county = name.split("_Short_Rain")[0].replace("_", " ").strip()
        return f"NDMA {county} Short Rains Assessment Report".strip()
    return f"NDMA {name.replace('_', ' ').strip()} Drought Early Warning Bulletin"


def extract_text_from_pdf(pdf_path: str, max_pages: int = 40) -> str:
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages[:max_pages]:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as exc:
        print(f"  [error reading {os.path.basename(pdf_path)}]: {exc}")
        return ""


def clean_and_split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    cleaned = []
    for s in raw_sentences:
        s = s.strip()
        words = s.split()
        if 6 <= len(words) <= 50:
            cleaned.append(s)
    return cleaned


def extract_all(max_pages: int) -> int:
    print("\n" + "=" * 70)
    print("STEP 3: EXTRACT")
    print("=" * 70)

    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, dtype=str)
    else:
        existing_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    seen_text = set(existing_df["English"].dropna().str.strip().str.lower())
    seen_files = set(existing_df["SourceFile"].dropna())

    new_rows = []
    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")) if os.path.exists(PDF_DIR) else []
    new_files = [f for f in pdf_files if f not in seen_files]
    print(f"{len(pdf_files)} PDFs on disk, {len(new_files)} not yet in {OUTPUT_CSV}")

    for idx, fname in enumerate(new_files):
        print(f"[{idx+1}/{len(new_files)}] extracting {fname}")
        text = extract_text_from_pdf(os.path.join(PDF_DIR, fname), max_pages=max_pages)
        source = derive_source(fname)
        for sentence in clean_and_split_sentences(text):
            key = sentence.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            new_rows.append({
                "English": sentence,
                "Domain": "Agriculture/Disaster",
                "Source": source,
                "SourceFile": fname,
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["English"], keep="first")
        combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

        os.makedirs(os.path.dirname(NEW_ROWS_FOR_MASTER_BUILD), exist_ok=True)
        new_df.to_csv(NEW_ROWS_FOR_MASTER_BUILD, index=False, encoding="utf-8")
        print(f"Also wrote {len(new_df)} new rows to {NEW_ROWS_FOR_MASTER_BUILD} "
              f"(build_master_datasets.py scans data/raw/ but not data/clean/, so this "
              f"is what actually feeds Master_PSA_Only.csv on the next build).")
    else:
        combined = existing_df

    print(f"\n{OUTPUT_CSV}: {len(existing_df)} -> {len(combined)} rows (+{len(combined) - len(existing_df)} new)")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Bulk-scrape NDMA KnowledgeWeb drought bulletins.")
    parser.add_argument("--skip-discover", action="store_true", help="reuse the existing manifest CSV")
    parser.add_argument("--skip-download", action="store_true", help="only re-run extraction on PDFs already on disk")
    parser.add_argument("--max-downloads", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between PDF downloads")
    parser.add_argument("--max-pages-per-pdf", type=int, default=40)
    parser.add_argument("--debug-dump", action="store_true", help="save raw HTML of crawled pages to data/raw/ndma_debug/")
    args = parser.parse_args()

    session = requests.Session()

    if args.skip_download:
        manifest = pd.DataFrame()
    elif args.skip_discover and os.path.exists(MANIFEST_PATH):
        manifest = pd.read_csv(MANIFEST_PATH, dtype=str)
    else:
        manifest = discover(session, args.debug_dump)

    if not args.skip_download:
        if manifest.empty:
            print("\nNo PDF URLs discovered. If this is unexpected, re-run with --debug-dump and "
                  "inspect data/raw/ndma_debug/*.html to see what the site actually returned -- "
                  "the category ID range or postback selectors likely need adjusting.")
        else:
            download_all(session, manifest, args.delay, args.max_downloads)

    total = extract_all(args.max_pages_per_pdf)
    print(f"\nFinal {OUTPUT_CSV} row count: {total}")


if __name__ == "__main__":
    main()
