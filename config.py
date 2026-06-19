
def get_config():
    training_path = "data/anime_training_data/train.jsonl"
    testing_path = "data/anime_training_data/test.jsonl"
    val_path = "data/anime_training_data/val.jsonl"
    label_map_path = "data/anime_training_data/label_map.json"
    save_path = "data/anime_training_data/bpe_tokenizer.json"


    id_to_label = {
        0: 'O', 
        1: 'B-LIKED_TITLE', 
        2: 'I-LIKED_TITLE', 
        3: 'B-POSITIVE_TERM', 
        4: 'I-POSITIVE_TERM', 
        5: 'B-NEGATIVE_TERM', 
        6: 'I-NEGATIVE_TERM'
    }

    label_to_id = {value: key for key, value in id_to_label.items()}
    return(training_path, testing_path, val_path, label_map_path, save_path, label_to_id)