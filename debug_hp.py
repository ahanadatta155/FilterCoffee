import os
import sys
import traceback

from scout.config import load_environment
from scout.tinyfish import TinyFishClient
from scout.clean import clean_page
from scout.represent import build_representation

def main():
    try:
        print("STEP 1: loading env", flush=True)
        load_environment()

        key = os.getenv("TINYFISH_API_KEY", "")
        print("STEP 2: TINYFISH_API_KEY present ->", bool(key), flush=True)
        if not key:
            print("NO KEY FOUND. Check your .env", flush=True)
            return

        url = "https://www.hp.com/in-en/shop/listings/laptops?campaignin=new"
        print("STEP 3: fetching ->", url, flush=True)

        client = TinyFishClient(key)
        p = client.fetch(url)
        print(f"STEP 4: fetched OK | bytes={len(p.html)} | tier={p.tier} | status={p.status_code} | final={p.url}", flush=True)

        cp = clean_page(p.html, p.url)
        print(f"STEP 5: clean_page() done | source_text={len(cp.source_text)} chars | main_text={len(cp.main_text)} chars | needs_render={cp.needs_render} | canvas_only={cp.canvas_only} | blocked={cp.blocked}", flush=True)

        rep = build_representation(cp)
        print(f"STEP 6: representation built | len={len(rep)} chars", flush=True)

        print("\n=== RAW HTML PREVIEW (first 400) ===", flush=True)
        print(p.html[:400], flush=True)

        print("\n=== SOURCE_TEXT SEEN BY MODEL (first 600) ===", flush=True)
        print(cp.source_text[:600], flush=True)

        print("\n=== REPRESENTATION (first 800) ===", flush=True)
        print(rep[:800], flush=True)
        print("\n=== END ===", flush=True)

    except Exception as e:
        print("CRASHED", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
