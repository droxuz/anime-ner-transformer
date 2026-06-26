import json
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/anime_dataset.csv", usecols = ["mal_id", "title", "title_english", "title_japanese"])
anime_list = df.copy()

def clean_title(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    return value if value else None


gazetteer = []

for _, row in anime_list.iterrows():
    title = clean_title(row["title"])
    title_english = clean_title(row["title_english"])
    title_japanese = clean_title(row["title_japanese"])

    # Prefer the standard title, then English, then Japanese.
    true_title = title or title_english or title_japanese

    if true_title is None:
        continue

    aliases = list(dict.fromkeys(
        alias
        for alias in [title, title_english, title_japanese]
        if alias is not None
    ))

    gazetteer.append({
        "mal_id": int(row["mal_id"]),
        "canonical_title": true_title,
        "aliases": aliases,
    })
output_path = Path("data/anime_training_data/title_gazetteer.json")
output_path.parent.mkdir(parents = True, exist_ok = True)
with output_path.open("w", encoding="UTF-8") as file:
    json.dump(gazetteer, file, ensure_ascii = False, indent = 2)

print(f"Saved new Gazetteer {len(gazetteer)}")