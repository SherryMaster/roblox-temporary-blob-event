"""Allow invocation with python -m blob_solver."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
