"""Allow running the CLI as ``python -m photo_album_scanner``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
