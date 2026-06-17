from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from exp2_word_vector_similarity import run


if __name__ == "__main__":
    print(run())
