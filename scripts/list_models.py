"""Discover models without sending any inference requests."""

import sys

from scout.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["models", *sys.argv[1:]]))
