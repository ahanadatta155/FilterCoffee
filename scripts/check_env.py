"""Report key presence without displaying keys or making API calls."""

import argparse
import os

from scout.config import load_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Explicit .env path; defaults to the project .env")
    args = parser.parse_args()
    load_environment(args.env_file)
    present = []
    for name in (
        "NEBIUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "NVIDIA_API_KEY",
    ):
        exists = bool(os.getenv(name, "").strip())
        present.append(exists)
        print(f"{name}: {'present' if exists else 'absent'}")
    tavily_status = "present" if os.getenv("TAVILY_API_KEY", "").strip() else "absent"
    print(f"TAVILY_API_KEY: {tavily_status}")
    tinyfish_status = "present" if os.getenv("TINYFISH_API_KEY", "").strip() else "absent"
    print(f"TINYFISH_API_KEY: {tinyfish_status}")
    return 0 if any(present) else 1


if __name__ == "__main__":
    raise SystemExit(main())
