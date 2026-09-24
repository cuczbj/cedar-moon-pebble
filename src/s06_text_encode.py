"""Standalone entry point; settings come from config.yaml."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == "__main__":
    sys.exit(main("s06_text_encode"))
