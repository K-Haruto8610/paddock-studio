"""Local inputs and race identity, shared by the patrol video entry points."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
FRAME = ROOT / "outputs/data-expansion-v20/frame_2020.pkl"
FINISH = ROOT / "outputs/v3/runs.pkl"
FORMATS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".mts", ".webm"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf8") as f:
        name = f.name
        try:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        except BaseException:
            f.close()
            Path(name).unlink()
            raise
    os.replace(name, path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def videos():
    folder = DATA / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in FORMATS)


def race_input(key):
    if not re.fullmatch(r"[0-9]{12}|[0-9]{16}", key):
        raise ValueError("Use RACE_CODE or YYYYMMDDVVRR (zero-padded).")
    d = pd.read_pickle(FRAME)
    codes = d.RACE_CODE.astype(str)
    chosen = d[codes == key] if len(key) == 16 else d[(codes.str[:10] + codes.str[-2:]) == key]
    if chosen.empty:
        # Recent declared fields may predate the next historical-frame refresh.
        pattern = key if len(key) == 16 else key[:10] + "????" + key[-2:]
        cards = list((ROOT/"data"/"entries").glob(pattern + ".json"))
        if len(cards) == 1:
            card = read_json(cards[0])
            race = card["race"]
            if card.get("status") == "declared" and not race.get("idProvisional", True) and card["raceId"] == cards[0].stem:
                numbers = [int(h["number"]) for h in card["horses"] if h["status"] != "scratched"]
                chosen = pd.DataFrame({"RACE_CODE": card["raceId"], "UMABAN": numbers, "field": len(numbers),
                    "venue": int(race["venueCode"]), "track": race["trackCode"], "distance": race["distance"]})
                for label in ["going", "soha", "last3", *(f"CORNER{j}_JUNI" for j in range(1, 5))]:
                    chosen[label] = float("nan")
    if chosen.RACE_CODE.nunique() != 1 or chosen.UMABAN.duplicated().any():
        raise ValueError(f"Missing, ambiguous or duplicate frame records for {key}")
    chosen = chosen.copy()
    chosen["UMABAN"] = chosen.UMABAN.astype(int)
    chosen = chosen.set_index("UMABAN").sort_index()
    if len(chosen) != int(chosen.field.iloc[0]):
        raise ValueError("The official frame does not cover the complete field.")
    code = str(chosen.RACE_CODE.iloc[0])
    candidates = [p for p in videos() if p.stem in {code, code[:10] + code[-2:]}]
    if len(candidates) != 1:
        raise ValueError(f"Expected one local video for {code}, found {len(candidates)}")
    return code, chosen, candidates[0]


def official_finish(code, horses):
    if not FINISH.is_file():
        return {}
    runs = pd.read_pickle(FINISH, compression="infer")
    ranks = runs.loc[runs.RACE_CODE == code].set_index("UMABAN").fin
    if not ranks.index.is_unique:
        raise ValueError("Duplicate official finish records")
    return ranks.reindex(horses).dropna().astype(int).to_dict()
