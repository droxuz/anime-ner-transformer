import json
import re
import unicodedata
from pathlib import Path


END_TOKEN = "__END__"


class AnimeGazetteer:
    def __init__(self, gazetteer_path):
        self.gazetteer_path = Path(gazetteer_path)
        self.trie = {}

        self._load_gazetteer()

    @staticmethod
    def tokenize(text):
        return re.findall(
            r"\w+(?:['’]\w+)*|[^\w\s]",
            text,
            flags=re.UNICODE,
        )

    @staticmethod
    def normalize_token(token):
        return unicodedata.normalize(
            "NFKC",
            token,
        ).casefold()

    def _load_gazetteer(self):
        with self.gazetteer_path.open("r",encoding="utf-8") as file:
            anime_entries = json.load(file)

        for anime in anime_entries:
            for alias in anime["aliases"]:
                alias_tokens = self.tokenize(alias)

                normalized_tokens = [
                    self.normalize_token(token)
                    for token in alias_tokens
                ]

                if not normalized_tokens:
                    continue

                self._insert_alias(
                    normalized_tokens,
                    anime={
                        "mal_id": anime["mal_id"],
                        "canonical_title": anime["canonical_title"],
                        "matched_alias": alias,
                    },
                )

    def _insert_alias(self, alias_tokens, anime):
        node = self.trie

        for token in alias_tokens:
            node = node.setdefault(token, {})

        terminal_entries = node.setdefault(END_TOKEN, [])

        # Avoid inserting the same MAL entry more than once.
        if not any(
            entry["mal_id"] == anime["mal_id"]
            for entry in terminal_entries
        ):
            terminal_entries.append(anime)

    def find_matches(self, tokens):
        """
        Find all possible title matches, then select the longest
        non-overlapping matches.

        end_token is exclusive.
        """
        normalized_tokens = [
            self.normalize_token(token)
            for token in tokens
        ]

        candidates = []

        for start in range(len(normalized_tokens)):
            node = self.trie

            for end in range(start, len(normalized_tokens)):
                current_token = normalized_tokens[end]

                if current_token not in node:
                    break

                node = node[current_token]

                if END_TOKEN in node:
                    candidates.append({
                        "start_token": start,
                        "end_token": end + 1,
                        "anime_candidates": node[END_TOKEN],
                    })

        # Prefer longer titles.
        candidates.sort(
            key=lambda match: (
                -(match["end_token"] - match["start_token"]),
                match["start_token"],
            )
        )

        selected_matches = []
        occupied = set()

        for match in candidates:
            start = match["start_token"]
            end = match["end_token"]

            match_positions = set(range(start, end))

            if occupied.intersection(match_positions):
                continue

            selected_matches.append(match)
            occupied.update(match_positions)

        return sorted(
            selected_matches,
            key=lambda match: match["start_token"],
        )

    def label_tokens(self, tokens):
        """
        Produce O, B-TITLE and I-TITLE labels.
        """
        labels = ["O"] * len(tokens)
        matches = self.find_matches(tokens)

        for match in matches:
            start = match["start_token"]
            end = match["end_token"]

            labels[start] = "B-TITLE"

            for index in range(start + 1, end):
                labels[index] = "I-TITLE"

        return labels, matches

    def label_prompt(self, prompt):
        tokens = self.tokenize(prompt)
        labels, matches = self.label_tokens(tokens)

        return {
            "tokens": tokens,
            "labels": labels,
            "matches": matches,
        }