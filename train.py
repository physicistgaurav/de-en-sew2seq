import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint
from attention_model import Encoder, Decoder, Seq2Seq, Attention

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Tokenizers
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Fields
german = Field(tokenize=tokenize_ger, lower=True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True,
                init_token="<sos>", eos_token="<eos>")

# Load Data
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english), root=".data"
)

# Build larger vocabulary
german.build_vocab(train_data, min_freq=2)
english.build_vocab(train_data, min_freq=2)

print(f"German vocabulary size: {len(german.vocab)}")
print(f"English vocabulary size: {len(english.vocab)}")

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

INPUT_DIM = len(german.vocab)
OUTPUT_DIM = len(english.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 20  # Increased for better convergence
BATCH_SIZE = 128
CLIP = 1

# Model, Optimizer, Loss
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
              DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# Initialize weights


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

writer = SummaryWriter("runs/loss_plot_seq2seq")
step = 0

# Training Functions


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # Turn off teacher forcing
            output = model(src, trg, 0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Training Loop
print("\nStarting training...")
print("="*50)

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
        save_checkpoint(checkpoint, filename="best_model.pth.tar")

    print(f"Epoch {epoch+1:02}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.3f} | "
          f"Val Loss: {valid_loss:.3f}")

# Load best model
print("\nLoading best model...")
checkpoint = torch.load("best_model.pth.tar", map_location=device)
model.load_state_dict(checkpoint["state_dict"])

# Final Evaluation
test_loss = evaluate(model, test_iterator, criterion)
print(f"\nTest Loss: {test_loss:.3f}")

# Example Translations
print("\n" + "="*50)
print("Example Translations:")
print("="*50)

example_sentences = [
    "ein mann in einem blauen hemd steht vor einem gelben hintergrund.",
    "eine frau geht die straße entlang.",
    "ein hund spielt im park.",
    "ich esse brot.",
    "das wetter ist schön."
]

for sentence in example_sentences:
    translation = translate_sentence(model, sentence, german, english, device)
    print(f"\nGerman: {sentence}")
    print(f"English: {' '.join(translation)}")

# BLEU Score
print("\n" + "="*50)
print("Calculating BLEU score on test set...")
bleu_score = bleu(test_data[:100], model, german,
                  english, device)  # Sample for speed
print(f"BLEU score: {bleu_score*100:.2f}")

# Save Final Checkpoint
checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": NUM_EPOCHS,
    "test_loss": test_loss
}
save_checkpoint(checkpoint, filename="my_checkpoint.pth.tar")
print("\n Final model saved successfully!")

# Interactive Testing
print("\n" + "="*50)
print("Training complete! Enter sentences to translate.")
print("Type 'quit' to exit.")
print("="*50)

model.eval()
while True:
    sentence = input("\nGerman sentence: ")
    if sentence.lower() == 'quit':
        break

    if not sentence.strip():
        continue

    translation = translate_sentence(model, sentence, german, english, device)
    print("Translated:", " ".join(translation))
