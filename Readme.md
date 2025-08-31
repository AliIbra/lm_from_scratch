
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

Please go over the notebooks in the defined orders, but you can run each component independently, inside each one of there is a specific requierment a comment will indicate the required :) 

### 1. Data Preparation

You need three text files: (different names)

* `train.txt`
* `validation.txt`
* `test.txt`

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


### 3. Running N-gram Models

### 4. Running Neural Trigram

### 5. Running Mini GPT

## Important Notes

* Run the code **cell by cell, in order**: BPE → N-gram → Neural Trigram → GPT.
* On **Colab**, make sure to upload or mount data first.
* On **local**, place text files in `./data/`.
* Training may take time — start with fewer epochs or merges for testing.
* Models can be **saved and reloaded** to continue training later.

---

## Observations

### 1. Effect of Number of BPE Merges

I tested multiple numbers of merges for BPE.

* With fewer than **2000 merges**, the **perplexity on validation text was too high**.
* Starting from around **4000 merges**, the perplexity began to flatten and stabilize.
* This is clearly visible in the following plot:

**My opinion**: The number of merges had a **huge impact** on the performance of all models.

---

### 2. Different BPE Variants for Different Models

I used more than one form of BPE depending on the model.

* This showed me that **data quality and variety** are crucial.
* But also the **quality of the tokenizer itself** matters:

  * Not only the model but also how we segment the data into subwords changes the results.


---

### 3. GPT Hyperparameters

Some key observations from GPT experiments:

* **Embedding size (`n_embd`)**: making it smaller speeds up training but reduces generation quality (eg. 128). 
* **Number of layers (`n_layer`)** and **heads (`n_head`)**: adding more helps performance, but training time grows quickly.
* **Batch size and sequence length**: larger values improve stability but need more memory.
* **Temperature & Top-k in generation**:

  * I didn't get a very much affect from playong with the temperature but with lower temperature, I got more deterministic output.
  * Top-k sampling helped prevent repetition.

**My opinion**: Hyperparameters can drastically change model behavior. Even small adjustments in **embedding size** or **temperature** make a noticeable difference in results.

---
