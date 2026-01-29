"""
Runner: build FDA warning letter panel.

- Inputs:  data/raw/warning_letters
- Outputs: data/processed/warning_letters/warning_letters_panel.feather
"""

from pathlib import Path
from src.prep_warning_letters import build_warning_letter_panel

def main() -> None:
    # Project root = parent of scripts/ (robust no matter where you run from)
    project_root = Path(__file__).resolve().parents[1]

    letter_path = project_root / "data" / "raw" / "warning_letters"
    output_file_path = project_root / "data" / "processed" / "warning_letters" / "warning_letters_panel.feather"

    # build the warning letter panel
    df = build_warning_letter_panel(
        letter_path=str(letter_path),
        output_file_path=str(output_file_path)
    )   

    print(f"Built warning letter panel with {len(df):,} rows.")

if __name__ == "__main__":
    main()