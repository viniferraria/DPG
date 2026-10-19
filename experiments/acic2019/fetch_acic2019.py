"""Download ACIC 2019 low-dimensional datasets for the DGPs used by ``run_acic2019.py``.

Fetches the official true-ATE table (maps each file to its DGP id) and the
200 MB low-dimensional archive, then extracts only the files of the requested
DGPs into ``datasets/dgp<id>/``. The archive is deleted afterwards. The data
is not redistributed in this repo.
"""

import argparse
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

TRUE_ATE_URL = "https://drive.google.com/uc?export=download&id=1GnZMX9tQp8-n1DRGsIf4DGGtO_3DrDu5"
LOW_DIM_URL = (
    "https://www.dropbox.com/scl/fi/fd9xwgcavesdp3vdj3mnx/low_dimensional_datasets.zip"
    "?rlkey=gvt9ku2y78j4sco1ko319crwm&dl=1"
)
DATASETS_DIR = Path(__file__).parent / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dgp", type=int, nargs="+", default=[64, 39])
    args = parser.parse_args()

    with urllib.request.urlopen(TRUE_ATE_URL) as resp:
        archive = io.BytesIO(resp.read())
    with zipfile.ZipFile(archive) as z:
        true_ate = pd.read_csv(z.open("lowDim_trueATE.csv"))
    wanted = true_ate[true_ate["DGPid"].isin(args.dgp)]
    missing = set(args.dgp) - set(wanted["DGPid"])
    if missing:
        raise SystemExit(f"DGP ids not in the low-dimensional track: {sorted(missing)}")

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        print("Downloading low-dimensional archive (~200 MB)...")
        with urllib.request.urlopen(LOW_DIM_URL) as resp:
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as z:
            for dgp_id, name in zip(wanted["DGPid"], wanted["filename"]):
                out = DATASETS_DIR / f"dgp{dgp_id}"
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{name}.csv").write_bytes(z.read(f"{name}.csv"))
    wanted.to_csv(DATASETS_DIR / "true_ate.csv", index=False)
    print(f"Wrote {len(wanted)} files under {DATASETS_DIR}")


if __name__ == "__main__":
    main()
