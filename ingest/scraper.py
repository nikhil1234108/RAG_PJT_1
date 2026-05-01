import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADER = {
    'User-Agent':(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data","articles")

def scrape_article(url:str) -> str:
    try:
        response = requests.get(url, headers=HEADER, timeout=35)
        response.raise_for_status()

    except Exception as e:
        print(f"[warn] failed to scrape {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["header", "footer", "nav", "aside", "script",
                     "style", "noscript", "form", "iframe"]):
        tag.decompose()

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    content_div = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="post-content") or
            soup.find("article")
    )

    body = content_div.get_text(separator=" ",strip=True) if content_div else ""

    return f"{title}\n\n{body}".strip()

def run_scraper(input_xlsx:str = "data/articles/Input.xlsx"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_xlsx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", input_xlsx)
    df = pd.read_excel(input_xlsx)
    print(f"found {len(df)} articles to scrape \n\n")

    for row in df.itertuples():
        url_id = str(row.URL_ID)
        url = str(row.URL)
        output_path = os.path.join(OUTPUT_DIR, f"{url_id}.txt")

        if os.path.exists(output_path):
            print(f"[skip] {url_id} already scraped")
            continue

        if not url.startswith("http"):
            print(f"[skip] {url_id} is not a valid url")
            continue

        print(f"[scraping] scraping {url_id}")

        try:
            text = scrape_article(url)

        except Exception as e:
            print(f"[scraping] failed to scrape {url}:{e}")

        if text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[scraped] {url_id} scraped")
        else:
            print(f"[scraped] {url_id} not scraped")

        time.sleep(1)

        print(f"\n Done scraping {url_id}")


if __name__ == "__main__":
    run_scraper()

