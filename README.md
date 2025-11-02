# German to English Neural Machine Translation

A seq2seq model with attention that translates German text to English. I built this to learn about attention mechanisms and how neural machine translation actually works under the hood.

This is a neural network that can translate German sentences into English. It uses an encoder-decoder architecture with Bahdanau attention - basically, the model learns to "focus" on different parts of the German sentence when generating each English word.
I trained it on the Multi30k dataset which has about 29,000 German-English sentence pairs. After 20 epochs of training, the model achieved a BLEU score of 27.53, which is decent for this architecture (modern Transformers do better, but this was a good learning experience).

## Tech Stacks

* PyTorch for building and training the neural network
* spaCy for tokenization (German and English)
* Multi30k dataset for training data


## How it works

* Encoder: A bidirectional GRU that reads the German sentence and creates hidden states
* Attention Mechanism: Helps the decoder figure out which parts of the input to focus on
* Decoder: Another GRU that generates the English translation word by word


## Installation

```
git clone https://github.com/physicistgaurav/de-en-sew2seq


create a virtual env

python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

```
python train.py
```
It saves the model as best_model.path.tar which will be used for testing.

BLEU Score achieved: 27.53 after running 20 epochs.

## MODEl Performance after 20 epochs:

* Train Loss: 0.710
* Validation Loss: 4.005
* LEU Score: 27.53


## Examples

Germany Sentence: ein mann in einem blauen hemd steht vor einem gelben hintergrund.
Target Output in English: A man in a blue shirt stands in front of a yellow background.
Output from model: a man in a blue shirt is standing in front of a yellow background .

Germany Sentence: Eine Frau in einer roten Jacke macht ein Foto von einem Hund, der über einen Parkweg rennt.
Target Output in English: A woman in a red jacket takes a picture of a dog running across a park path.
Output from model: a woman in a red jacket is taking a photo of a dog .

Germany Sentence: Drei Kinder spielen lachend Fußball auf einem offenen Feld neben einem kleinen Wald.
Target Output in English: Three children are laughing as they play football in an open field next to a small forest.
Output from model: three children are playing soccer ball on a field with a a a a .

## Images

![Screenshots](/images/epochs.png "20/20 epochs")
![Screenshots](/images/example.png "test screenshots")
