
def get_config():
    training_path = "data/anime_training_data/train.jsonl"
    testing_path = "data/anime_training_data/test.jsonl"
    val_path = "data/anime_training_data/val.jsonl"
    label_map_path = "data/anime_training_data/label_map.json"
    save_path = "data/anime_training_data/bpe_tokenizer.json"


    id_to_label = {
        0: 'O',
        1: 'B-TITLE',
        2: 'I-TITLE',
        3: 'B-GENRE',
        4: 'I-GENRE',
        5: 'B-THEME',
        6: 'I-THEME',
    }

    label_to_id = {value: key for key, value in id_to_label.items()}
    return(training_path, testing_path, val_path, label_map_path, save_path, label_to_id)
