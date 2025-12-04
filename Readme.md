# 🌐 Attention Is All You Need — English ↔ Hindi Machine Translation

A complete PyTorch implementation of the Transformer architecture applied to the IITB English–Hindi machine translation task. The project includes sentencepiece-based BPE tokenization, dataset preparation, model training, and inference.

---

## 📁 Project Structure

.</br>
</br>├── Transformer/
</br>│ ├── components.py
</br>│ ├── encoder.py
</br>│ ├── decoder.py
</br>│ └── transformer.py
</br>│
</br>├── Tokenizer/
</br>│ └── tokenizer.py
</br>│
</br>├── Data/
</br>│ └── dev_test/
</br>│ ├── dev.en
</br>│ ├── dev.hi
</br> ├── test.en
</br>│ ├── test.hi
</br>│ ├── dev.all
</br>│ ├── test.all
</br>│ ├── bpe.vocab
</br>│ └── ...
</br>│
</br>├── train.py
</br>├── inference.py
</br>├── dev.ipynb
</br>├── loading_collab.ipynb
</br>└── README.md

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
- `ParallelTextDataset`  
- `collate_fn` (handles dynamic padding & mask creation)  
- Training loop with teacher forcing  

### `dev.ipynb`
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
python inference.py --text "Hello world"
