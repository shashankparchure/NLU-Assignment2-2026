"""
scrape_iitj_firstpart.py

First part of the scraping pipeline for the IIT Jodhpur Word2Vec assignment.
This script will:
 - crawl the IIT Jodhpur site (iitj.ac.in) for specific sections: People (faculty lists), news, academic, and downloads
 - follow links to individual faculty profiles and external profile pages when present
 - download and extract text from HTML pages and PDFs (regulations, syllabi)
 - save a raw corpus file: iitj_corpus_raw.txt

USAGE:
  python scrape_iitj_firstpart.py

Note: This is the first part (crawler + downloader + extractor). It writes iitj_corpus_raw.txt.
Make sure dependencies are installed (see bottom of file).

"""

import requests
from bs4 import BeautifulSoup
import time
import re
import os
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path

# PDF extraction libraries
# We'll try using pdfminer.six (recommended) and fall back to textract if needed
from pdfminer.high_level import extract_text as pdf_extract_text

# ========== CONFIG ==========
BASE_DOMAIN = "iitj.ac.in"
BASE_URL = "https://www.iitj.ac.in/"
OUTPUT_DIR = Path("./scrape_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_CORPUS_PATH = OUTPUT_DIR / "iitj_corpus_raw.txt"
DOWNLOADS_DIR = OUTPUT_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# polite crawling
REQUESTS_SLEEP = 0.8  # seconds between requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IITJ-Scraper/1.0; +https://example.com)"
}

# limiters
MAX_FACULTY_PROFILES = 15  # user requested 15 faculty pages

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ========== HELPERS ==========

def is_same_domain(url):
    try:
        p = urlparse(url)
        return p.netloc.endswith(BASE_DOMAIN)
    except Exception:
        return False


def safe_request(url, max_retries=2, timeout=15):
    for attempt in range(max_retries+1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            time.sleep(REQUESTS_SLEEP)
            return r
        except Exception as e:
            logging.warning(f"Request failed ({attempt}) for {url}: {e}")
            time.sleep(1 + attempt)
    logging.error(f"Giving up on {url}")
    return None


def clean_text_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")

    # remove scripts and styles
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def download_and_extract_pdf(url, save_dir=DOWNLOADS_DIR):
    logging.info(f"Downloading PDF: {url}")
    r = safe_request(url)
    if r is None:
        return ""

    # create a filename
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path) or "doc.pdf"
    local_path = save_dir / fname
    with open(local_path, "wb") as f:
        f.write(r.content)

    # try to extract text via pdfminer
    try:
        text = pdf_extract_text(str(local_path)) or ""
        text = text.strip()
        logging.info(f"Extracted {len(text)} chars from PDF {fname}")
        return text
    except Exception as e:
        logging.error(f"PDF extract failed for {fname}: {e}")
        return ""


# ========== CRAWL STRATEGY ==========
# We'll target the main useful pages on iitj.ac.in and then follow faculty profile links.
# Key entry points (live queries will discover the exact URLs):
ENTRY_POINTS = [
    urljoin(BASE_URL, "People"),           # faculty listing
    urljoin(BASE_URL, "news"),             # news listing
    urljoin(BASE_URL, "announcements"),    # announcements / circulars
    urljoin(BASE_URL, "Academic"),        # academic pages (regulations, curriculum)
    urljoin(BASE_URL, "Computer-Science-Engineering"),
    urljoin(BASE_URL, "Mechanical-Engineering"),
    urljoin(BASE_URL, "Electrical-Engineering"),
    urljoin(BASE_URL, "School-of-Liberal-Arts"),
]

# We'll gather found faculty profile links in this list (unique)
faculty_links = []
discovered_links = set()
collected_texts = []


def extract_links_from_listing(page_url):
    """Fetch page, find internal profile links (hrefs under /People/ or /People/Profile or dept pages)."""
    r = safe_request(page_url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # normalize
        if href.startswith("/"):
            href_full = urljoin(BASE_URL, href.lstrip('/'))
        elif href.startswith("http"):
            href_full = href
        else:
            href_full = urljoin(page_url, href)

        # keep only iitj domain or likely profile pages
        if "People" in href_full or "/Profile/" in href_full or BASE_DOMAIN in urlparse(href_full).netloc:
            links.append(href_full)
    return links


def gather_faculty_profiles(limit=MAX_FACULTY_PROFILES):
    logging.info("Gathering faculty profile links from entry points...")
    global faculty_links

    for ep in ENTRY_POINTS:
        logging.info(f"Scanning {ep}")
        links = extract_links_from_listing(ep)
        for l in links:
            if len(faculty_links) >= limit:
                break
            # simple heuristic: profile links contain 'People' or 'Profile' or 'faculty'
            if l not in discovered_links and ("People" in l or "/Profile/" in l or "/faculty" in l.lower() or "/People/List" in l):
                discovered_links.add(l)
                faculty_links.append(l)
        if len(faculty_links) >= limit:
            break

    logging.info(f"Found {len(faculty_links)} faculty-related links (limit {limit})")
    return faculty_links


def scrape_profile_or_page(url):
    # follow the page: if it links externally to a web profile (e.g., scholar.iitj.ac.in or research.iitj.ac.in), follow that too
    logging.info(f"Scraping page: {url}")
    r = safe_request(url)
    if r is None:
        return ""

    content_type = r.headers.get('Content-Type', '')
    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
        return download_and_extract_pdf(url)

    text = clean_text_from_html(r.text)

    # find other profile links to follow
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        if href.startswith('mailto:'):
            continue
        # normalize
        if href.startswith("/"):
            href_full = urljoin(BASE_URL, href.lstrip('/'))
        elif href.startswith("http"):
            href_full = href
        else:
            href_full = urljoin(url, href)

        parsed = urlparse(href_full)
        # follow profiles which are either on iitj.ac.in or scholar.iitj.ac.in or other institute-managed domains
        if any(domain in parsed.netloc for domain in [BASE_DOMAIN, "scholar.iitj.ac.in", "faculty.iitj.ac.in", "research.iitj.ac.in"]):
            if href_full not in discovered_links and len(discovered_links) < 500:
                logging.info(f"Following linked profile: {href_full}")
                discovered_links.add(href_full)
                r2 = safe_request(href_full)
                if r2 is not None:
                    # if PDF
                    if 'application/pdf' in r2.headers.get('Content-Type','') or href_full.lower().endswith('.pdf'):
                        pdf_text = download_and_extract_pdf(href_full)
                        text += "\n" + pdf_text
                    else:
                        text += "\n" + clean_text_from_html(r2.text)

    return text


# ========== MAIN RUN ==========

def main():
    logging.info("START: IITJ scraping first part")

    # 1) Gather faculty profile links
    profiles = gather_faculty_profiles()

    # If we didn't find enough via the entry points, try the People listing root
    if len(profiles) < MAX_FACULTY_PROFILES:
        logging.info("Trying People root page fallback")
        profiles += extract_links_from_listing(urljoin(BASE_URL, "People/List"))
        profiles = list(dict.fromkeys(profiles))[:MAX_FACULTY_PROFILES]

    # 2) Crawl each profile and collect text
    count = 0
    for p in profiles:
        if count >= MAX_FACULTY_PROFILES:
            break
        try:
            txt = scrape_profile_or_page(p)
            if txt and len(txt) > 20:
                collected_texts.append(txt)
                count += 1
        except Exception as e:
            logging.error(f"Error scraping {p}: {e}")

    # 3) Also try to fetch academic regulation PDFs from the Academic page
    academic_page = urljoin(BASE_URL, "Academic")
    logging.info(f"Scanning academic page for PDFs: {academic_page}")
    r = safe_request(academic_page)
    if r:
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                pdf_url = urljoin(academic_page, href)
                pdf_text = download_and_extract_pdf(pdf_url)
                if pdf_text:
                    collected_texts.append(pdf_text)

    # 4) Save raw corpus
    logging.info(f"Saving raw corpus to {RAW_CORPUS_PATH}")
    with open(RAW_CORPUS_PATH, "w", encoding="utf-8") as f:
        for doc in collected_texts:
            # simple separator
            f.write(doc + "\n\n###DOC_BREAK###\n\n")

    logging.info("Finished first part scraping. Output written.")


if __name__ == '__main__':
    main()


# ========== DEPENDENCIES & NOTES ==========
# pip install requests beautifulsoup4 pdfminer.six
# On some systems, pdfminer may struggle; alternative: textract (requires external utils)
# If you get blocked by rate limits, reduce REQUESTS_SLEEP or run during off-peak hours.
# The script intentionally follows internal profile links and small external institute-managed domains.
# This is the first part: crawling + download + raw corpus save. Next steps: cleaning, tokenization, Word2Vec training.