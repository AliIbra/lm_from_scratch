# Language Modeling from Scratch: BPE → N-gram → Neural Trigram → Mini GPT

## Overview

This repo contains different stages of building a language model:

1. **BPE Tokenizer**
2. **N-gram Models**
3. **Neural Trigram Model**
4. **Mini GPT (Transformer)**

Each stage is provided in a separate script / notebook cell.
To run everything, just follow the steps below either **locally** or in **Google Colab**.

---

## Running the Code

### 1. Data Preparation

You need three text files:

* `train.txt`
* `validation.txt`
* `test.txt`

These should contain plain text (Shakespeare dataset).

#### On Colab

Upload your files interactively:

```python
from google.colab import files
uploaded = files.upload()  # Choose train.txt, validation.txt, test.txt
```

Or copy from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
DATA_FOLDER = "/content/drive/MyDrive/data"
TRAIN_FILE = f"{DATA_FOLDER}/train.txt"
VALIDATION_FILE = f"{DATA_FOLDER}/validation.txt"
TEST_FILE = f"{DATA_FOLDER}/test.txt"
```

#### Locally

Make a `data/` folder in your repo and place your files:

```
data/train.txt
data/validation.txt
data/test.txt
```

Then in Python:

```python
DATA_FOLDER = "./data"
TRAIN_FILE = f"{DATA_FOLDER}/train.txt"
VALIDATION_FILE = f"{DATA_FOLDER}/validation.txt"
TEST_FILE = f"{DATA_FOLDER}/test.txt"
```

---

### 2. Running BPE

Run the BPE cell first:

```python
train_text = load_text(TRAIN_FILE)
vocab, merges = byte_pair_encoding(train_text, num_merges=2000)
tokens = bpe_tokenize_word("hello", merges)
print(tokens)
```

---

### 3. Running N-gram Models

After BPE, run the N-gram cell:

```python
ngram_counts = count_ngrams(train_text, n=3, merges=merges)
ngram_probs = calculate_ngram_probabilities(ngram_counts, 3)

from collections import Counter
vocab_size = len(get_subword_token_vocab(vocab))
smoothed_probs = laplace_smoothing(ngram_counts, ngram_probs, vocab_size)

print(generate_text(smoothed_probs, 3, start_token='h', length=50))
```

---

### 4. Running Neural Trigram

After BPE + n-grams, run the Neural Trigram cell:

```python
model = NeuralTrigramModel(vocab_size=len(token_to_id))
train_model(model, inputs, targets, epochs=20, lr=0.05)
print(generate_text(model, token_to_id, id_to_token, start_tokens=['the']))
```

---

### 5. Running Mini GPT

Finally, run the Transformer (Mini GPT): (it is preferred to run on Colab) 

```python
model = GPT(config).to(DEVICE)
train(model, train_loader, val_loader, iters=500)
save_model(model, "/content/drive/MyDrive/colab_models/gpt_model.pt")
```

For text generation:

```python
generate_text(model, "Once upon a time", max_new_tokens=50, temperature=0.8, top_k=40)
```

---

## Important Notes

* Run the code cell by cell, in order: BPE → N-gram → Neural Trigram → GPT.
* On **Colab**, make sure to upload or mount data first.
* On **local**, place text files in `./data/`.
* Training may take time — start with fewer epochs or merges for testing.
* Models can be **saved and reloaded** to continue training later.




