Diverse v2 anime prompt training data
Generated from uploaded train(1).jsonl, val(1).jsonl, test(1).jsonl.
The new splits use different prompt templates by split, held-out title pools for val/test, and extra natural surface forms.

train: {"rows": 50000, "avg_len": 13.20158, "max_len": 33, "examples_with_liked": 31641, "examples_with_positive": 33388, "examples_with_negative": 27828, "empty_examples": 3, "labels": {"O": 329421, "I-LIKED_TITLE": 122088, "B-POSITIVE_TERM": 59854, "B-NEGATIVE_TERM": 50777, "B-LIKED_TITLE": 36113, "I-POSITIVE_TERM": 33517, "I-NEGATIVE_TERM": 28309}}

val: {"rows": 7000, "avg_len": 16.766285714285715, "max_len": 36, "examples_with_liked": 4167, "examples_with_positive": 4448, "examples_with_negative": 4213, "empty_examples": 2, "labels": {"O": 72026, "I-LIKED_TITLE": 16661, "B-POSITIVE_TERM": 8041, "B-NEGATIVE_TERM": 7324, "B-LIKED_TITLE": 4716, "I-POSITIVE_TERM": 4442, "I-NEGATIVE_TERM": 4154}}

test: {"rows": 7000, "avg_len": 15.422285714285714, "max_len": 36, "examples_with_liked": 4200, "examples_with_positive": 4591, "examples_with_negative": 4547, "empty_examples": 2, "labels": {"O": 62051, "I-LIKED_TITLE": 16399, "B-POSITIVE_TERM": 8125, "B-NEGATIVE_TERM": 7677, "B-LIKED_TITLE": 4812, "I-POSITIVE_TERM": 4604, "I-NEGATIVE_TERM": 4288}}

Point config.py to this folder to train on the regenerated data.
