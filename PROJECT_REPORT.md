# Project Report: Attention Is All You Need

## 1. Introduction

This project is a from-scratch implementation of the Transformer architecture for English-to-Hindi and Hindi-to-English style sequence modeling, centered on the ideas introduced in the paper *Attention Is All You Need*. Based on the repository `Readme.md`, the main objective is to build a complete machine translation pipeline using PyTorch, SentencePiece tokenization, parallel corpus preprocessing, model training, checkpointing, evaluation, and inference.

The repository is not just a model definition. It is a full workflow project that includes:

- dataset preparation utilities,
- tokenization and vocabulary generation,
- byte-offset-based large file preprocessing,
- Transformer encoder-decoder implementation,
- distributed training,
- evaluation with loss, perplexity, and BLEU,
- command-line inference,
- and a Streamlit-based translation interface with MongoDB logging.

The `README` is the best high-level source for understanding the intended structure and goals of the project, while the code shows the latest implementation state, especially for the UI and persistence features added later.

## 2. Problem Statement

The project addresses the problem of neural machine translation between English and Hindi using the Transformer architecture. Traditional recurrent architectures struggle with long-range dependencies and limited parallelism during training. The Transformer solves this by replacing recurrence with attention mechanisms, allowing efficient parallel computation and stronger context modeling.

In this repository, the practical problem can be stated as follows:

> Build an end-to-end English-Hindi machine translation system that can preprocess bilingual data, tokenize it efficiently, train a Transformer model on parallel text, evaluate its quality, and serve translations through an interactive application.

The implementation focuses on the IITB English-Hindi parallel corpus described in the `README`, and the codebase is designed to support the full development lifecycle from experimentation to deployment-facing inference.

## 3. Project Scope and Goals

From the `README` and the implementation, the main goals of the project are:

- reproduce the core Transformer architecture in PyTorch from first principles,
- train the model on English-Hindi parallel data,
- use SentencePiece BPE to handle open-vocabulary translation,
- support scalable preprocessing through tokenized binary datasets and byte offsets,
- support multi-GPU or distributed execution through DDP,
- evaluate model behavior with cross-entropy, perplexity, and BLEU,
- provide inference both from the terminal and through a Streamlit UI,
- and store user-preferred translation outputs using MongoDB.

## 4. Repository Overview

The top-level repository is organized around the translation pipeline:

- `Transformer/`: core architecture, encoder, decoder, attention, and checkpoint utilities.
- `Tokenizer/`: SentencePiece training, loading, and encoding/decoding helpers.
- `Dataset/`: tokenized dataset reader and dynamic batch padding.
- `ByteOffsetGenerator/`: utilities for creating offset files and tokenized binary corpora.
- `DDP/`: distributed training setup and cleanup.
- `Logger/`: logging configuration for training runs.
- `Score/`: BLEU score implementation.
- `train.py`: distributed training pipeline.
- `test.py`: evaluation pipeline for loss, perplexity, and BLEU.
- `inference.py`: checkpoint loading and translation generation.
- `streamlit_app.py`: user interface for translation and preference submission.
- `db/Mongo.py`: MongoDB connection and write logic for selected outputs.
- `Dev/` and `viz/`: experimentation notebooks and training visualizations.
- `Refrences/Attentionpaper2017.pdf`: foundational research reference used by the project.

## 5. Core Implementation

## 5.1 Tokenization Strategy

The tokenization layer is implemented in `Tokenizer/tokenizer.py` using SentencePiece BPE.

Important implementation details:

- The tokenizer either loads an existing `bpe.model` or trains a new one if loading fails.
- Vocabulary size is set to `16000`.
- Model type is `bpe`.
- Extra encoding options automatically add BOS and EOS tokens.
- Control symbols `<start>` and `<end>` are also defined.

Special token IDs documented in the `README` are:

- `<pad>` = 0
- `<unk>` = 1
- `<bos>` = 2
- `<eos>` = 3
- `<start>` = 4
- `<end>` = 5

This is a strong design choice for machine translation because subword tokenization helps represent both English and Hindi without requiring an impractically large word-level vocabulary.

## 5.2 Dataset Preparation and Storage

The project avoids expensive line-by-line text loading at training time by converting data into binary token arrays with offset files.

Two parts support this:

- `ByteOffsetGenerator/script.py` creates byte offsets for source and target text files.
- `Tokenizer/pretoken.py` uses those offsets and the SentencePiece tokenizer to generate:
  - `.tok.bin` token storage files
  - `.tok.idx` token offset files

The actual training dataset reader is implemented in `Dataset/parallelDataSet.py`.

Its main responsibilities are:

- loading integer token arrays for English and Hindi,
- loading sentence boundary offsets,
- returning aligned sentence pairs as tensors,
- and padding each batch dynamically via `collate_fn`.

This design is efficient for larger corpora because it minimizes repeated text parsing and allows direct indexed access into tokenized data.

## 5.3 Transformer Architecture

The Transformer implementation is located in the `Transformer/` package and is the technical center of the repository.

### `Transformer/components.py`

This file defines:

- `Attention`: multi-head attention logic using `scaled_dot_product_attention`,
- `FFN`: the position-wise feed-forward network,
- `AddNorm`: a custom normalization layer used in the model blocks.

One notable implementation choice is that attention uses:

- separate query projection,
- shared key/value style projection behavior,
- flash/scaled dot-product attention when available,
- and explicit masked attention support for decoder self-attention.

### `Transformer/encoder.py`

This file implements:

- `EncoderLayer`,
- stacked `Encoder`,
- and a custom dropout module.

Each encoder block performs:

1. pre-normalization,
2. self-attention,
3. residual connection,
4. feed-forward transformation,
5. residual connection,
6. final encoder normalization after all layers.

### `Transformer/decoder.py`

This file implements:

- `DecoderLayer`,
- stacked `Decoder`,
- and a matching custom dropout module.

Each decoder block performs:

1. masked self-attention,
2. encoder-decoder cross-attention,
3. feed-forward processing,
4. residual connections around each sublayer,
5. final decoder normalization.

### `Transformer/transformer.py`

This file assembles the full model.

Important features:

- learnable token embeddings,
- learnable positional embeddings,
- separate encoder and decoder stacks,
- output projection to vocabulary logits,
- pad mask generation,
- training mode with teacher forcing,
- inference mode with beam-search-style decoding.

Key hyperparameters used in training and inference are:

- embedding dimension: `512`
- feed-forward dimension: `2048`
- attention heads: `8`
- encoder layers: `6`
- decoder layers: `6`
- max tokens: `256`

These values are aligned with a standard full Transformer-style setup and reflect a serious implementation rather than a toy example.

## 5.4 Training Pipeline

Training is implemented in `train.py`.

The training pipeline includes:

- distributed training initialization via `DDP/ddp.py`,
- model wrapping with `DistributedDataParallel`,
- mixed precision support using `GradScaler`,
- label smoothing,
- gradient clipping,
- custom learning-rate scheduling,
- periodic checkpoint saving,
- automatic checkpoint resume,
- and structured logging.

Important training details found in the code:

- optimizer: Adam with `betas=(0.9, 0.98)` and `eps=1e-8`
- learning rate schedule: inverse square root schedule with warmup
- warmup steps: `4000`
- label smoothing: `0.01`
- epochs: `5`
- batch size: `100` in the current script

The learning-rate schedule directly mirrors the schedule popularized by the original Transformer paper:

- scale by `d_model^-0.5`
- use the minimum of inverse square root decay and warmup growth

This is one of the clearest signs that the project attempts to stay close to the original research design rather than only using default PyTorch settings.

## 5.5 Checkpointing and Recovery

Checkpoint logic is handled in `Transformer/checkpoint.py`.

The implementation supports:

- timestamped checkpoint names,
- saving model, optimizer, scaler, step, and epoch state,
- loading the latest checkpoint automatically,
- local mode and Colab/runtime copy mode.

This is an important engineering feature because long training runs are fragile without restart support.

## 5.6 Evaluation Pipeline

Model evaluation is implemented in `test.py`.

The script computes:

- test loss,
- perplexity,
- BLEU score.

BLEU is implemented locally in `Score/BLEU.py`. The implementation computes:

- clipped n-gram precision up to order `n`,
- smoothing for zero counts,
- brevity penalty,
- final BLEU aggregation.

This shows that the project does not stop at training loss and includes a translation-quality metric that is standard in machine translation work.

## 5.7 Inference Pipeline

Inference is implemented in `inference.py`.

The inference flow is:

1. build tokenizer,
2. build Transformer with the same architecture used for training,
3. find the newest checkpoint in `saves/`,
4. load model weights,
5. tokenize the input sentence,
6. run autoregressive decoding,
7. remove special tokens,
8. decode generated tokens back into Hindi text.

The current implementation exposes:

- `load_model()` for reusable loading,
- `translate_text()` for programmatic inference,
- a command-line `main()` function for interactive usage.

The model currently returns multiple beam outputs in the translation path, and the Streamlit app uses those as user-selectable options.

## 5.8 Streamlit Application and MongoDB Integration

The latest visible functional expansion in the repository is the user-facing application implemented in `streamlit_app.py`.

This layer adds:

- a browser-based interface for English input,
- cached model loading,
- display of checkpoint path and compute device,
- multiple Hindi translation options,
- user selection of the preferred translation,
- persistence of the selected option to MongoDB.

Database persistence is implemented in `db/Mongo.py`.

That module provides:

- environment-driven MongoDB configuration,
- URI normalization for credentials,
- TLS configuration with `certifi`,
- database and collection resolution,
- explicit connection validation via `ping`,
- insertion of user feedback records,
- and user-friendly error handling for TLS issues.

This is a meaningful step beyond a classroom-only implementation because it introduces application feedback capture, which can later support qualitative evaluation, human preference collection, or dataset enrichment.

## 6. Entire Development Path

The repository history, notebook artifacts, and saved outputs together show a fairly clear development path.

### Phase 1: Initial implementation and dataset preparation

The earliest commit history from December 2025 shows:

- initial Transformer implementation,
- README creation,
- dataset loading notebook creation,
- DataLoader preparation,
- Colab-based experimentation.

This suggests the project started as a research/learning build developed interactively, with notebooks used to understand the dataset and prototype the training flow.

### Phase 2: Training infrastructure and checkpoint support

Commits from December 4, 2025 onward show work on:

- checkpointing,
- dropout and label smoothing,
- making the training path closer to the original paper,
- preparing the system to run on Colab and local environments.

This phase reflects the move from a conceptual prototype to a trainable system.

### Phase 3: Architecture refinement

Commits from mid-December indicate:

- shared KV attention work,
- transition from post-norm toward pre-norm behavior,
- final normalization in encoder and decoder,
- server and local execution improvements.

This phase appears focused on stability and architectural correctness.

### Phase 4: Data efficiency and debugging

Late December and early January additions include:

- byte-offset generation,
- tokenized binary dataset support,
- visualization notebooks for per-epoch or per-step loss debugging,
- checkpoint cleanup and repository organization improvements.

The presence of `viz/training_metrics.csv`, `viz/viz.ipynb`, and the training log indicates that the author actively debugged unstable training behavior and used metrics to improve convergence.

### Phase 5: Completion of core training and evaluation

By January 9, 2026, the commit history explicitly mentions that the implementation was completed. The repository also contains:

- a saved checkpoint in `saves/`,
- evaluation code in `test.py`,
- BLEU scoring,
- and a polished `README`.

This phase marks the completion of the end-to-end machine translation pipeline.

### Phase 6: Productization and interface layer

The most recent major phase appears in late March 2026:

- codebase refactoring,
- Streamlit UI integration,
- MongoDB-backed selection storage,
- improved project usability for interactive translation.

This changes the project from a pure model-training repository into a lightweight application.

## 7. Evidence of Training and Progress

The repository contains a training checkpoint:

- `saves/Transformer_step_16000_2026-01-09_09-42-09.pt`

The training log in `logs/train.log` shows:

- multiple restarts and resumptions,
- batch size changes across runs,
- reduction of loss over time,
- checkpoint-based continuation.

This provides practical evidence that the model was not only written but actually trained and iterated on.

## 8. Strengths of the Implementation

The strongest technical qualities of the repository are:

- clear separation of concerns across tokenizer, dataset, model, training, evaluation, and inference,
- faithful implementation of major Transformer training ideas,
- practical support for checkpointing and recovery,
- use of binary token storage for efficiency,
- distributed training support,
- evaluation beyond loss alone,
- user-facing interface for translation demonstration,
- and persistence of human preference data through MongoDB.

Another strength is that the project shows both research thinking and engineering thinking. The notebooks and references show exploration, while the scripts and packaging show movement toward reproducibility and usage.

## 9. Observations and Current-State Notes

Because this report considers both the `README` and the current implementation, a few important observations should be noted:

- The `README` is the best high-level source, but it does not fully capture the latest Streamlit and MongoDB additions.
- The current codebase mixes research-style experimentation with application-style features.
- There are some naming inconsistencies such as `Readme.md` vs. `README.md` conventions and `Refrences` spelling.
- The project depends on existing trained artifacts like `bpe.model`, `bpe.vocab`, and saved checkpoints for full inference use.
- `pyproject.toml` shows a recent packaging step with `streamlit`, `pymongo`, `python-dotenv`, `sentencepiece`, and `torch`.

These do not weaken the project. They mainly show that it evolved organically from experimentation into a more complete application.

## 10. References

The following repository sources were most useful for reconstructing the report:

- `Readme.md` as the primary project narrative and structure guide.
- `Transformer/transformer.py`, `Transformer/encoder.py`, `Transformer/decoder.py`, and `Transformer/components.py` for the architecture.
- `train.py` for the end-to-end training procedure.
- `Dataset/parallelDataSet.py` and `Tokenizer/pretoken.py` for dataset handling.
- `Tokenizer/tokenizer.py` for the SentencePiece setup.
- `test.py` and `Score/BLEU.py` for evaluation.
- `inference.py` for runtime translation flow.
- `streamlit_app.py` and `db/Mongo.py` for the user application and persistence layer.
- `Transformer/checkpoint.py`, `Logger/logger.py`, and `DDP/ddp.py` for engineering utilities.
- `logs/train.log`, `saves/`, `Dev/`, and `viz/` for development evidence.

External conceptual reference:

- Vaswani et al., *Attention Is All You Need* (2017), included in the repository at `Refrences/Attentionpaper2017.pdf`.

## 11. Conclusion

This repository is a substantial end-to-end Transformer-based machine translation project built around English-Hindi translation. It begins with raw bilingual data, transforms that data into efficient tokenized training inputs, trains a custom PyTorch Transformer with distributed and mixed-precision support, evaluates the model using standard translation metrics, and finally exposes the system through both script-based inference and a Streamlit application.

The project’s development path shows a realistic evolution:

- initial experimentation,
- architecture implementation,
- training stabilization,
- efficiency improvements,
- evaluation support,
- and finally interactive deployment.

Overall, this is not only an implementation of the Transformer paper but also a practical machine translation system that demonstrates model building, data engineering, experimentation, evaluation, and basic application integration in one repository.
