#!/usr/bin/env python3
"""Download arxiv papers as PDFs from root_papers.txt."""

import os
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FILE = "/home/ubuntu/EVO/dataset/root_papers.txt"
OUTPUT_DIR = "/home/ubuntu/EVO/dataset/pdf"


def sanitize_filename(title):
    """Remove or replace characters that are unsafe for filenames."""
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = re.sub(r'\s+', '_', title.strip())
    title = title.strip('_.')
    if len(title) > 150:
        title = title[:150]
    return title


def download_paper(arxiv_id, title, output_dir):
    """Download a single paper from arxiv."""
    safe_title = sanitize_filename(title)
    filename = f"{arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return arxiv_id, "skipped (exists)"

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (research paper downloader)"
            })
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
                if len(data) < 1000:
                    return arxiv_id, f"failed (too small: {len(data)} bytes)"
                with open(filepath, 'wb') as f:
                    f.write(data)
                return arxiv_id, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
                continue
            elif e.code == 404:
                return arxiv_id, "failed (404 not found)"
            else:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                return arxiv_id, f"failed (HTTP {e.code})"
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return arxiv_id, f"failed ({e})"

    return arxiv_id, "failed (max retries)"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    papers = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arxiv_id, title = line.split(',', 1)
            papers.append((arxiv_id.strip(), title.strip()))

    total = len(papers)
    print(f"Downloading {total} papers to {OUTPUT_DIR}")

    ok = 0
    skipped = 0
    failed = 0
    failed_list = []

    # Use limited concurrency + delay to avoid rate limiting
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, (arxiv_id, title) in enumerate(papers):
            future = executor.submit(download_paper, arxiv_id, title, OUTPUT_DIR)
            futures[future] = (arxiv_id, title)
            # Small stagger to avoid burst
            if i % 4 == 3:
                time.sleep(1)

        for future in as_completed(futures):
            arxiv_id, status = future.result()
            if status == "ok":
                ok += 1
            elif "skipped" in status:
                skipped += 1
            else:
                failed += 1
                failed_list.append((arxiv_id, status))

            done = ok + skipped + failed
            if done % 20 == 0 or done == total:
                print(f"  Progress: {done}/{total} (ok={ok}, skipped={skipped}, failed={failed})")

    print(f"\nDone! ok={ok}, skipped={skipped}, failed={failed}")
    if failed_list:
        print("\nFailed papers:")
        for aid, status in failed_list:
            print(f"  {aid}: {status}")


if __name__ == "__main__":
    main()
