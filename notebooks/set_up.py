"""
Shared setup for notebooks.

Repo layout:

data/
  demo/
  private/
  synth_data/

results/
  output/
    validation/
    cate_estimator/
    study_design/
  tables/
  figures/

Nothing under data/private/ should be committed.
"""

# notebooks/set_up.py
from pathlib import Path


def _find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking upward until expected folders are found."""
    cur = (start or Path.cwd()).resolve()
    while True:
        if (cur / "src").exists() and (cur / "data").exists() and (cur / "notebooks").exists():
            return cur
        if cur.parent == cur:
            raise RuntimeError(
                "Could not locate repo root (missing src/, data/, notebooks/)."
            )
        cur = cur.parent


REPO_ROOT = _find_repo_root()

# --- Data directories ---
DATA_DIR = REPO_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
PRIVATE_DIR = DATA_DIR / "private"
SYNTH_DIR = DATA_DIR / "synth_data"

# --- Results directories ---
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "output"
VALIDATION_DIR = OUTPUT_DIR / "validation"
CATE_DIR = OUTPUT_DIR / "cate_estimator"
STUDY_DESIGN_DIR = OUTPUT_DIR / "study_design"

TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create result folders if they do not exist."""
    for p in [
        RESULTS_DIR,
        OUTPUT_DIR,
        VALIDATION_DIR,
        CATE_DIR,
        STUDY_DESIGN_DIR,
        TABLE_DIR,
        FIG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
