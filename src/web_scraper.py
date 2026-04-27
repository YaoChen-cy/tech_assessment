# Scrape and download WM investor relations news-release PDFs using Playwright.
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright


def _sanitize_filename(name: str) -> str:
    # Replace filesystem-unsafe characters and spaces in a filename.
    name = re.sub(r'[\\/*?:"<>|]', "_", name.strip())
    return name.replace(" ", "_")


def _is_news_release(title: str, aria: str, text: str) -> bool:
    # Return True if the link attributes identify a news-release PDF.
    combined = f"{title} {aria} {text}".lower()
    return "news release" in combined


class WMNewsReleaseScraper:
    def __init__(self, base_url: str, page_url: str, save_dir: str):
        # Store base URL, results page URL, and local save directory.
        self.base_url = base_url
        self.page_url = page_url
        self.save_dir = save_dir

    def _collect_pdf_links(self, page) -> list[tuple[str, str]]:
        # Return list of (url, filename) for news-release PDFs on the page.
        page.wait_for_selector("a[type='application/pdf']", timeout=30000)

        results, seen = [], set()
        for a in page.query_selector_all("a[type='application/pdf']"):
            href  = a.get_attribute("href") or ""
            title = a.get_attribute("title") or ""
            aria  = a.get_attribute("aria-label") or ""
            text  = a.inner_text()

            if not href or not _is_news_release(title, aria, text):
                continue

            url      = urljoin(self.base_url, href)
            filename = _sanitize_filename(title or text + ".pdf")

            if url in seen:
                continue
            seen.add(url)
            results.append((url, filename))

        return results

    def _run_in_playwright(self) -> list[str]:
        # Launch a Playwright browser, scrape PDF links, and download each file.
        saved_paths = []
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-http2", "--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = context.new_page()

            print(f"Loading {self.page_url} ...")
            page.goto(self.page_url, wait_until="load", timeout=60000)

            pdfs = self._collect_pdf_links(page)
            print(f"Found {len(pdfs)} news release PDFs.")

            for i, (url, filename) in enumerate(pdfs, 1):
                save_path = os.path.join(self.save_dir, filename)
                print(f"[{i}/{len(pdfs)}] Downloading {filename}")
                try:
                    response = context.request.get(
                        url,
                        headers={"Referer": self.page_url, "Accept": "application/pdf,*/*"},
                        timeout=180000,
                    )
                    if not response.ok:
                        print(f"  HTTP {response.status}: {url}")
                        continue
                    with open(save_path, "wb") as f:
                        f.write(response.body())
                    print(f"  Saved: {save_path}")
                    saved_paths.append(save_path)
                    time.sleep(1)
                except Exception as e:
                    print(f"  Failed: {url}\n  Reason: {e}")

            browser.close()

        return saved_paths

    def run(self) -> list[str]:
        # Scrape and download all news-release PDFs; return saved file paths.
        os.makedirs(self.save_dir, exist_ok=True)
        # Run in a thread so sync_playwright doesn't conflict with Jupyter's event loop.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(self._run_in_playwright).result()


if __name__ == "__main__":
    scraper = WMNewsReleaseScraper(
        base_url="https://investors.wm.com",
        page_url="https://investors.wm.com/financials/financial-results",
        save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw/news_releases"),
    )
    scraper.run()
