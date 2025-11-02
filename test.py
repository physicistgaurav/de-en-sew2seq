import torch
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field
from utils import load_checkpoint, translate_sentence
from attention_model import Encoder, Decoder, Seq2Seq, Attention

# Tokenizers
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Rebuild Fields and Vocab
german = Field(tokenize=tokenize_ger, lower=True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True,
                init_token="<sos>", eos_token="<eos>")

# Load data to rebuild vocab (must match training)
train_data, _, _ = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english), root=".data"
)

german.build_vocab(train_data, min_freq=2)
english.build_vocab(train_data, min_freq=2)

# Hyperparameters (must match training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(german.vocab)
OUTPUT_DIM = len(english.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Build Model Architecture
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(encoder, decoder, device).to(device)

#  Load Checkpoint
try:
    checkpoint = torch.load("best_model.pth.tar", map_location=device)
    load_checkpoint(checkpoint, model, optimizer=None)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: best_model.pth.tar not found!")
    print("Please run 'python train.py' first to train and save the model.")
    exit(1)

print("=" * 50)

#  Interactive Translation
model.eval()

print("Enter German sentences to translate.")
print("Type 'quit' to exit.\n")

while True:
    sentence = input("German: ")

    if sentence.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if not sentence.strip():
        continue

    translation = translate_sentence(model, sentence, german, english, device)
    print(f"English: {' '.join(translation)}\n")
