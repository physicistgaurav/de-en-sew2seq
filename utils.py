import torch
import spacy
from torchtext.data.metrics import bleu_score

spacy_ger = spacy.load("de_core_news_sm")


def translate_sentence(model, sentence, german, english, device, max_length=50):
    """
    Translate a single German sentence using the trained model with attention.
    """
    if isinstance(sentence, str):
        tokens = [tok.text.lower() for tok in spacy_ger(sentence)]
    else:
        tokens = [tok.lower() for tok in sentence]

    tokens = [german.init_token] + tokens + [german.eos_token]
    text_to_indices = [german.vocab.stoi.get(
        tok, german.vocab.stoi["<unk>"]) for tok in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(sentence_tensor)

        outputs = [english.vocab.stoi["<sos>"]]

        for _ in range(max_length):
            prev_word = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, hidden = model.decoder(
                    prev_word, hidden, encoder_outputs)
                top_idx = output.argmax(1).item()

            outputs.append(top_idx)

            if top_idx == english.vocab.stoi["<eos>"]:
                break

    translated_tokens = [english.vocab.itos[idx] for idx in outputs[1:]]
    if translated_tokens and translated_tokens[-1] == "<eos>":
        translated_tokens = translated_tokens[:-1]

    return translated_tokens


def bleu(data, model, german, english, device, max_len=50):
    """
    Compute corpus BLEU score over a torchtext Example dataset (train/val/test).
    """
    model.eval()
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        # this is a list of tokens (lowercased by the Field)
        trg = vars(example)["trg"]
        prediction = translate_sentence(
            model, src, german, english, device, max_length=max_len)

        # Ensure we remove any trailing <eos> from prediction
        if len(prediction) and prediction[-1] == "<eos>":
            prediction = prediction[:-1]

        # bleu_score expects a list of references per example
        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None, lr=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # optional: override lr
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
