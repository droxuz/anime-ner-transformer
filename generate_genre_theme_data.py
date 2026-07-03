import json
import random
import re
from collections import Counter
from pathlib import Path


OUTPUT_DIR = Path("data/anime_training_data")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|'[A-Za-z]+|[^\w\s]")
SEED = 321

GENRE_SURFACES = {
    "action": ["action", "action heavy", "fight scenes", "combat focused", "battle heavy"],
    "adventure": ["adventure", "journey", "quest", "exploration", "adventure vibes"],
    "comedy": ["comedy", "funny", "jokes", "light comedy", "goofy"],
    "drama": ["drama", "character drama", "relationship drama", "emotional drama"],
    "fantasy": ["fantasy", "magic", "swords and magic", "magical worlds", "dark fantasy"],
    "horror": ["horror", "scary", "creepy", "ghost stories", "monster horror"],
    "mystery": ["mystery", "detective story", "puzzle solving", "investigation"],
    "romance": ["romance", "love story", "romantic drama", "dating", "romantic stuff"],
    "sci-fi": ["sci-fi", "science fiction", "space opera", "future tech"],
    "slice of life": ["slice of life", "everyday life", "quiet daily life"],
    "sports": ["sports", "sports anime", "team sports", "competition sports"],
    "supernatural": ["supernatural", "spirits", "ghosts", "supernatural powers"],
    "mecha": ["mecha", "giant robots", "robot shows", "mech suits"],
    "isekai": ["isekai", "transported to another world", "other world fantasy"],
    "martial arts": ["martial arts", "hand to hand fights", "dojo battles"],
    "military": ["military", "war drama", "army setting", "tactical battles"],
    "historical": ["historical", "historical setting", "period piece"],
    "psychological": ["psychological", "mind games", "mental games", "psychological twists"],
    "thriller": ["thriller", "suspense", "tense mystery", "high tension"],
    "music": ["music", "band story", "idol music", "performing arts"],
}

THEME_SURFACES = {
    "school life": ["school life", "high school", "academy setting", "classroom setting", "student cast"],
    "adult cast": ["adult cast", "older cast", "grown up cast", "adult characters"],
    "coming of age": ["coming of age", "growing up", "self discovery"],
    "found family": ["found family", "chosen family", "team bonding"],
    "revenge": ["revenge", "payback", "vengeance plot"],
    "survival": ["survival", "desperate survival", "staying alive"],
    "time travel": ["time travel", "timeline loops", "changing the past"],
    "tournament": ["tournament", "tournament arc", "ranked battles"],
    "political intrigue": ["political intrigue", "court politics", "power struggles"],
    "war": ["war", "battlefield trauma", "wartime choices"],
    "grief": ["grief", "loss", "mourning", "tearjerker"],
    "redemption": ["redemption", "second chances", "atonement"],
    "friendship": ["friendship", "strong friendships", "crew loyalty"],
    "betrayal": ["betrayal", "backstabbing", "broken trust"],
    "dark tone": ["dark tone", "bleak atmosphere", "grim mood"],
    "wholesome tone": ["wholesome", "comforting tone", "soft healing"],
    "mature tone": ["mature tone", "serious tone", "grown up tone"],
    "slow pacing": ["slow pacing", "slow burn", "patient pacing"],
    "fast pacing": ["fast pacing", "quick pacing", "keeps moving"],
    "violence": ["gore", "blood", "bloody violence", "graphic violence"],
}

FALLBACK_TITLES = [
    "Cowboy Bebop",
    "Fullmetal Alchemist: Brotherhood",
    "Steins;Gate",
    "Attack on Titan",
    "Violet Evergarden",
    "Mushishi",
    "Kaguya-sama: Love is War",
    "Haikyuu!!",
    "Mob Psycho 100",
    "Nana",
]


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text


def clean_title(value):
    value = str(value).strip()
    if not value or any(ord(character) > 127 for character in value):
        return None
    if len(tokenize(value)) > 10:
        return None
    return value


def load_titles():
    path = OUTPUT_DIR / "title_gazetteer.json"
    if not path.exists():
        return FALLBACK_TITLES

    with path.open("r", encoding="utf-8") as file:
        gazetteer = json.load(file)

    titles = []
    seen = set()
    for row in gazetteer:
        for alias in row.get("aliases", [row.get("canonical_title", "")]):
            title = clean_title(alias)
            normalized = title.casefold() if title else None
            if title and normalized not in seen:
                titles.append(title)
                seen.add(normalized)

    return titles or FALLBACK_TITLES


def split_title_pools(titles, rng):
    titles = titles[:6000]
    rng.shuffle(titles)
    train_end = int(len(titles) * 0.70)
    val_end = int(len(titles) * 0.85)
    return {
        "train": titles[:train_end],
        "val": titles[train_end:val_end],
        "test": titles[val_end:],
    }


def choose_surface(surface_map, rng):
    canonical = rng.choice(list(surface_map))
    surface = rng.choice(surface_map[canonical])
    return canonical, surface


def append_segment(tokens, labels, text, label_type=None):
    segment_tokens = tokenize(text)
    if not segment_tokens:
        return

    tokens.extend(segment_tokens)
    if label_type is None:
        labels.extend(["O"] * len(segment_tokens))
        return

    labels.append(f"B-{label_type}")
    labels.extend([f"I-{label_type}"] * (len(segment_tokens) - 1))


def build_row(segments, metadata):
    tokens = []
    labels = []
    for text, label_type in segments:
        append_segment(tokens, labels, text, label_type)

    title_terms = extract_terms(tokens, labels, "TITLE")

    return {
        "prompt": detokenize(tokens),
        "tokens": tokens,
        "labels": labels,
        "liked_titles": title_terms,
        "title_terms": title_terms,
        "genre_terms": metadata["genres"],
        "theme_terms": metadata["themes"],
        "wanted_genres": metadata["wanted_genres"],
        "avoided_genres": metadata["avoided_genres"],
        "wanted_themes": metadata["wanted_themes"],
        "avoided_themes": metadata["avoided_themes"],
    }


def extract_terms(tokens, labels, label_type):
    terms = []
    current = []

    for token, label in zip(tokens, labels):
        if label == f"B-{label_type}":
            if current:
                terms.append(detokenize(current))
            current = [token]
        elif label == f"I-{label_type}" and current:
            current.append(token)
        else:
            if current:
                terms.append(detokenize(current))
                current = []

    if current:
        terms.append(detokenize(current))

    return terms


def entity_pair(rng, title_pool):
    title = rng.choice(title_pool)
    genre_a, genre_surface_a = choose_surface(GENRE_SURFACES, rng)
    genre_b, genre_surface_b = choose_surface(GENRE_SURFACES, rng)
    theme_a, theme_surface_a = choose_surface(THEME_SURFACES, rng)
    theme_b, theme_surface_b = choose_surface(THEME_SURFACES, rng)
    return {
        "title": title,
        "genre_a": genre_a,
        "genre_surface_a": genre_surface_a,
        "genre_b": genre_b,
        "genre_surface_b": genre_surface_b,
        "theme_a": theme_a,
        "theme_surface_a": theme_surface_a,
        "theme_b": theme_b,
        "theme_surface_b": theme_surface_b,
    }


def metadata_from_slots(slots, wanted_genres=(), avoided_genres=(), wanted_themes=(), avoided_themes=()):
    return {
        "titles": [slots["title"]],
        "genres": list(dict.fromkeys([*wanted_genres, *avoided_genres])),
        "themes": list(dict.fromkeys([*wanted_themes, *avoided_themes])),
        "wanted_genres": list(wanted_genres),
        "avoided_genres": list(avoided_genres),
        "wanted_themes": list(wanted_themes),
        "avoided_themes": list(avoided_themes),
    }


def make_examples(title_pool, rng):
    slots = entity_pair(rng, title_pool)
    title = slots["title"]
    genre_a = slots["genre_surface_a"]
    genre_b = slots["genre_surface_b"]
    theme_a = slots["theme_surface_a"]
    theme_b = slots["theme_surface_b"]
    gca = slots["genre_a"]
    gcb = slots["genre_b"]
    tca = slots["theme_a"]
    tcb = slots["theme_b"]

    examples = [
        (
            [("I liked ", None), (title, "TITLE"), ("; give me ", None), (genre_a, "GENRE"), (" with ", None), (theme_a, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("No ", None), (genre_a, "GENRE"), (", no ", None), (theme_a, "THEME"), (", but ", None), (genre_b, "GENRE"), (" is fine.", None)],
            metadata_from_slots(slots, wanted_genres=[gcb], avoided_genres=[gca], avoided_themes=[tca]),
        ),
        (
            [("Something like ", None), (title, "TITLE"), (" without the ", None), (genre_a, "GENRE"), (" angle; keep the ", None), (theme_a, "THEME"), (".", None)],
            metadata_from_slots(slots, avoided_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("I am chasing ", None), (theme_a, "THEME"), (" and ", None), (genre_a, "GENRE"), (", not ", None), (theme_b, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca], avoided_themes=[tcb]),
        ),
        (
            [("After ", None), (title, "TITLE"), (", I want more ", None), (genre_a, "GENRE"), ("; avoid ", None), (genre_b, "GENRE"), (" if possible.", None)],
            metadata_from_slots(slots, wanted_genres=[gca], avoided_genres=[gcb]),
        ),
        (
            [("Can you find a ", None), (genre_a, "GENRE"), (" show where ", None), (theme_a, "THEME"), (" matters?", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("\"", None), (title, "TITLE"), ("\" worked for me. More ", None), (theme_a, "THEME"), (", less ", None), (theme_b, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_themes=[tca], avoided_themes=[tcb]),
        ),
        (
            [("I would watch ", None), (genre_a, "GENRE"), (" even if it has ", None), (theme_a, "THEME"), (", but skip ", None), (genre_b, "GENRE"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca], avoided_genres=[gcb]),
        ),
        (
            [("Please recommend ", None), (genre_a, "GENRE"), (" / ", None), (genre_b, "GENRE"), (" with ", None), (theme_a, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca, gcb], wanted_themes=[tca]),
        ),
        (
            [("The title I remember is ", None), (title, "TITLE"), (", and the part I want is ", None), (theme_a, "THEME"), (" rather than ", None), (genre_a, "GENRE"), (".", None)],
            metadata_from_slots(slots, avoided_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("Not picky on art, just make it ", None), (genre_a, "GENRE"), (" with ", None), (theme_a, "THEME"), (" and some ", None), (theme_b, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca, tcb]),
        ),
        (
            [("I bounced off ", None), (title, "TITLE"), (", but I still like ", None), (genre_a, "GENRE"), (" and ", None), (theme_a, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("Anything tagged as ", None), (genre_a, "GENRE"), (" is okay; anything centered on ", None), (theme_a, "THEME"), (" is not.", None)],
            metadata_from_slots(slots, wanted_genres=[gca], avoided_themes=[tca]),
        ),
        (
            [("I want ", None), (theme_a, "THEME"), (" in a ", None), (genre_a, "GENRE"), (" shell, similar to ", None), (title, "TITLE"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("Hard pass on ", None), (genre_a, "GENRE"), (" unless the show has ", None), (theme_a, "THEME"), (" like ", None), (title, "TITLE"), (".", None)],
            metadata_from_slots(slots, avoided_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("Give me something that starts ", None), (genre_a, "GENRE"), (" but turns into ", None), (theme_a, "THEME"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca]),
        ),
        (
            [("Looking for ", None), (genre_a, "GENRE"), (", ", None), (theme_a, "THEME"), (", and no ", None), (theme_b, "THEME"), (" at all.", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca], avoided_themes=[tcb]),
        ),
        (
            [("The vibe of ", None), (title, "TITLE"), (" plus ", None), (genre_a, "GENRE"), (" would be ideal.", None)],
            metadata_from_slots(slots, wanted_genres=[gca]),
        ),
        (
            [("Keep ", None), (genre_a, "GENRE"), ("; drop the ", None), (theme_a, "THEME"), (" and ", None), (genre_b, "GENRE"), (".", None)],
            metadata_from_slots(slots, wanted_genres=[gca], avoided_genres=[gcb], avoided_themes=[tca]),
        ),
        (
            [("Is there a ", None), (genre_a, "GENRE"), (" anime about ", None), (theme_a, "THEME"), (" that is not just ", None), (genre_b, "GENRE"), ("?", None)],
            metadata_from_slots(slots, wanted_genres=[gca], wanted_themes=[tca], avoided_genres=[gcb]),
        ),
    ]

    return examples


def generate_split(split_name, size, title_pool, seed_offset):
    rng = random.Random(SEED + seed_offset)
    rows = []

    while len(rows) < size:
        for segments, metadata in make_examples(title_pool, rng):
            rows.append(build_row(segments, metadata))
            if len(rows) >= size:
                break

    rng.shuffle(rows)
    return rows


def validate_rows(rows):
    allowed = {"O", "B-TITLE", "I-TITLE", "B-GENRE", "I-GENRE", "B-THEME", "I-THEME"}
    for index, row in enumerate(rows):
        if len(row["tokens"]) != len(row["labels"]):
            raise ValueError(f"row {index} token/label length mismatch")
        unknown = set(row["labels"]) - allowed
        if unknown:
            raise ValueError(f"row {index} has unknown labels {sorted(unknown)}")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lexicon(path):
    payload = {
        "genres": GENRE_SURFACES,
        "themes": THEME_SURFACES,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def summarize(rows):
    labels = Counter(label for row in rows for label in row["labels"])
    lengths = [len(row["tokens"]) for row in rows]
    return {
        "rows": len(rows),
        "avg_len": round(sum(lengths) / len(lengths), 2),
        "max_len": max(lengths),
        "labels": dict(sorted(labels.items())),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    title_pools = split_title_pools(load_titles(), rng)
    sizes = {"train": 72000, "val": 12000, "test": 12000}

    summaries = {}
    for offset, split_name in enumerate(("train", "val", "test")):
        rows = generate_split(split_name, sizes[split_name], title_pools[split_name], offset)
        validate_rows(rows)
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        summaries[split_name] = summarize(rows)

    write_lexicon(OUTPUT_DIR / "genre_theme_lexicon.json")
    with (OUTPUT_DIR / "README.txt").open("w", encoding="utf-8") as file:
        file.write("Genre/theme BIO anime prompt training data\n")
        file.write("Labels: O, B-TITLE, I-TITLE, B-GENRE, I-GENRE, B-THEME, I-THEME\n")
        file.write("Generated by generate_genre_theme_data.py with held-out title pools by split.\n\n")
        for split_name in ("train", "val", "test"):
            file.write(f"{split_name}: {json.dumps(summaries[split_name], ensure_ascii=False)}\n")

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
