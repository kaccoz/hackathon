"""ACV prediction entry point: --input FILE_OR_FOLDER --output CSV."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from baseline import create_predictions, main

if __name__ == "__main__":
    main()
