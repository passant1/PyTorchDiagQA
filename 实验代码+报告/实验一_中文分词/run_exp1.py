from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from exp1_ffnn_segmentation import run


if __name__ == "__main__":
    print(run())
