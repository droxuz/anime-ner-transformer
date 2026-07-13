from dataset import load_jsonl
from config import get_config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from pathlib import Path

#Loads Config
training_path, testing_path, val_path, label_map_path, save_path, label_to_id = get_config()
mal_synopsis_path = "data/anime_training_data/mal_synopsis.jsonl"
pretrained_bpe_path = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"

#data is a dictionary of "prompt", "tokens", "labels"
#training the BPE using the data
def train_bpe_tokenizer(data, save_path, vocab_size = 50000):

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Normalize data, Pre-tokenize Keep the Capitalization for similar to BERT case
    # tokenizer.normalizer = Sequence([NFKC(),Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    #Define training
    trainer = BpeTrainer( min_frequency=2, special_tokens=["<PAD>", "<UNK>"], initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator(data, trainer= trainer)
    tokenizer.save(save_path)

    return tokenizer

def load_bpe_tokenizer(save_path):
    return Tokenizer.from_file(save_path)

def get_bpe_tokenizer(data, save_path, vocab_size=50000, force_retrain=False):
    tokenizer_path = Path(save_path)

    if tokenizer_path.exists() and not force_retrain:
        return load_bpe_tokenizer(save_path)

    return train_bpe_tokenizer(data= data, save_path= save_path)

# synopsis_data = load_jsonl(mal_synopsis_path)
# BPE_tokens = train_bpe_tokenizer(synopsis_data, pretrained_bpe_path)

tokenizer = load_bpe_tokenizer(pretrained_bpe_path)
# print("Actual vocab size:", tokenizer.get_vocab_size())

test_titles = ["Attack on Titan", "Steins;Gate", "Re:Zero", "Fullmetal Alchemist"]
for title in test_titles:
    count = 0
    encoded = tokenizer.encode(title)
    print(f"{title:30} → {encoded.tokens}")
    for token in encoded.tokens:
        count += 1 
    print(f"Tokens #: {count}")
