# 🌐 Attention Is All You Need — English ↔ Hindi Machine Translation

A complete PyTorch implementation of the Transformer architecture applied to the IITB English–Hindi machine translation task. The project includes sentencepiece-based BPE tokenization, dataset preparation, model training, and inference.

---

## 📁 Project Structure

.</br>
</br>├── Transformer/
</br>│ ├── checkpoint.py
</br>│ ├── components.py
</br>│ ├── encoder.py
</br>│ ├── decoder.py
</br>│ └── transformer.py
</br>│
</br>├── Tokenizer/
</br>│ ├── pretoken.py
</br>│ └── tokenizer.py
</br>│
</br>├── Dataset/
</br>│ └── parallelDataSet.py
</br>│
</br>├── DDP/
</br>│ └── ddp.py
</br>│
</br>├── Data/
</br>│ ├── dev_test/
</br>│ ├── parallel-n/
</br>│ └── tokenized/
</br>│
</br>├── ByteOffsetGenerator/
</br>│ ├── script.py
</br>│ ├── en_offset.bo
</br>│ └── hi_offset.bo
</br>│
</br>├── Logger/
</br>│ └── logger.py
</br>│
</br>├── viz/
</br>│ ├── viz.ipynb
</br>│ └── training_metrics.csv
</br>│
</br>├── Dev/
</br>│ ├── dev.ipynb
</br>│ ├── MOE.ipynb
</br>│ └── lr_scheduing.ipynb
</br>│
</br>├── Refrences/
</br>│ └── Attentionpaper2017.pdf
</br>│
</br>├── logs/
</br>│ └── train.log
</br>│
</br>├── saves/
</br>│ └── (Model Checkpoints)
</br>│
</br>├── train.py
</br>├── test.py
</br>├── inference.py
</br>├── loading_collab.ipynb
</br>├── bpe.model
</br>├── bpe.vocab
</br>├── logs.txt
</br>└── Readme.md

---

## 🚀 Transformer Architecture

This project contains a from-scratch PyTorch re-implementation of the architecture described in *Attention Is All You Need* (Vaswani et al., 2017).

### 🔧 Core Modules (`Transformer/`)

#### **components.py**
- **Attention**  
  - Multi-head attention  
  - Scaled dot-product: `softmax(QKᵀ / √d_k) V`  
  - Supports self-attention, cross-attention, and masked decoding attention  
- **FFN**  
  - Two-layer position-wise feed-forward network  

#### **encoder.py**
Implements the **EncoderLayer**:
- Multi-head self-attention  
- Feed-forward network  
- Residual connections  
- Layer normalization  

#### **decoder.py**
Implements the **DecoderLayer**:
- Masked self-attention  
- Encoder–decoder cross-attention  
- Feed-forward network  
- Residual + LayerNorm after each block  

#### **transformer.py**
Defines the complete Transformer model:
- Token embeddings  
- Positional embeddings  
- Stacked Encoder and Decoder  
- Key methods:  
  - `embed()`  
  - `make_pad_mask()`  
  - `encode()`  
  - `decode()`  

---

## 🔤 Tokenization & Vocabulary — SentencePiece BPE

The project uses **SentencePiece** with **BPE (Byte-Pair Encoding)**.

### Special Tokens
| Token       | ID |
|-------------|----|
| `<pad>`     | 0  |
| `<unk>`     | 1  |
| `<bos>`     | 2  |
| `<eos>`     | 3  |
| `<start>`   | 4  |
| `<end>`     | 5  |

### `Tokenizer/tokenizer.py`
- Trains BPE model on combined `.all` corpus  
- Encodes/decodes text  
- Stores vocabulary (`bpe.vocab`)  
- Used by training & inference pipelines  

---

## 📊 Dataset — IITB English–Hindi Corpus

Located in `Data/dev_test/`.

Files include:
- `dev.en`, `dev.hi`  
- `test.en`, `test.hi`  
- `dev.all`, `test.all`  
- BPE vocabulary  

### `loading_collab.ipynb`
- Loads IITB dataset  
- Preprocesses & merges files  
- Saves cleaned versions  

---

## 🏋️ Training

### `train.py`
Includes:
- Training loop with teacher forcing  
- Handles distributed data parallel (DDP) logging, saving, and evaluation.

### `Dataset/parallelDataSet.py`
- `ParallelTextDataset`
- `collate_fn` (handles dynamic padding & mask creation)

### `Dev/dev.ipynb`
Used for:
- Experimentation (e.g., embedding_dims = 12, n_heads = 3, d_ff = 48)  
- Debugging model  
- Running small-scale tests  

---

## 🔍 Inference

### `inference.py`
- Loads trained model  
- Encodes input using SentencePiece  
- Autoregressive decoding until `<end>` token  
- Detokenizes output  

Run:
```bash
python inference.py
