from dataset import load_jsonl
from config import get_config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, NFKC, Lowercase

#Loads Config
training_path, testing_path, val_path, label_map_path, save_path, label_to_id = get_config()

#data is a dictionary of "prompt", "tokens", "labels"
#training the BPE using the data
def train_bpe_tokenizer(data, save_path, vocab_size = 8000):

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Normalize data, Pre-tokenize, 
    tokenizer.normalizer = Sequence([NFKC(),Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    #Define training
    trainer = BpeTrainer(vocab_size=8000, min_frequency=2, special_tokens=["<PAD>", "<UNK>"], initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator((row["prompt"] for row in data), trainer= trainer)
    tokenizer.save(save_path)

    return tokenizer

def load_bpe_tokenizer(save_path):
    return Tokenizer.from_file(save_path)

data = load_jsonl(training_path)
BPE_tokens = train_bpe_tokenizer(data, save_path, vocab_size= 8000)
