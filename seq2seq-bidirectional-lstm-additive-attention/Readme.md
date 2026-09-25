# Seq2Seq + Bidirectional LSTM + Additive Attention

## English → French Neural Machine Translation

This project implements an **English-to-French Neural Machine Translation (NMT)** system using a **Sequence-to-Sequence architecture with a Bidirectional LSTM encoder and custom Additive Attention mechanism**.

The model is trained using the **Helsinki-NLP OPUS Books English–French dataset** available through Hugging Face. The original dataset contains 127,085 translation pairs, from which up to **100,000 clean and usable sentence pairs** are selected for the experiment. The selected dataset is divided into **80,000 training, 10,000 validation, and 10,000 test samples**.
The system uses a **256-dimensional embedding**, a **256-unit Bidirectional LSTM encoder**, a **512-unit decoder LSTM**, and a **custom Bahdanau-style Additive Attention layer**. The attention mechanism explicitly handles encoder padding rather than relying on automatic Keras mask propagation.

### Key Features

* English → French neural machine translation
* Hugging Face OPUS Books dataset
* Up to 100,000 translation pairs
* 80/10/10 train-validation-test split
* TextVectorization-based tokenization
* Sequence-to-Sequence architecture
* Bidirectional LSTM encoder
* LSTM decoder
* Custom Additive/Bahdanau Attention
* Teacher forcing
* Padding-aware masked loss
* Padding-aware token accuracy
* AdamW optimization
* Dropout and L2 regularization
* Gradient clipping
* Learning-rate scheduling
* Early stopping
* Best-model checkpointing
* Greedy decoding
* Temperature-based sampling
* BLEU-4 evaluation
* Token-level confusion matrix
* Attention heatmap visualization
* Qualitative translation comparison
* Model weights and vocabulary/configuration saving

The notebook is designed as a complete **Deep Learning / NLP / Sequence Modeling portfolio project**, demonstrating not only model training but also inference, evaluation, visualization, and model artifact management.

# 🌍 English → French Neural Machine Translation

## Seq2Seq + Bidirectional LSTM + Additive Attention

A Deep Learning based **Neural Machine Translation (NMT)** project that translates English sentences into French using a **Sequence-to-Sequence model with a Bidirectional LSTM encoder, LSTM decoder, and custom Additive Attention mechanism**.

The project uses the **Helsinki-NLP OPUS Books English–French dataset** from Hugging Face and trains on up to **100,000 filtered translation pairs**.

---

## 📌 Project Overview

Neural Machine Translation aims to automatically convert text from one natural language into another.

In this project:

```text
English Sentence
       ↓
Text Cleaning
       ↓
Text Vectorization
       ↓
Embedding
       ↓
Bidirectional LSTM Encoder
       ↓
Encoder Hidden Representations
       ↓
LSTM Decoder
       ↓
Additive Attention
       ↓
Dense + Softmax
       ↓
French Sentence
```

The model learns the relationship between English source sequences and French target sequences using **teacher forcing during training**.

## The notebook loads the OPUS Books `en-fr` configuration from Hugging Face. The available raw dataset contains **127,085 English–French pairs**, and the experiment selects up to **100,000 usable pairs** after cleaning and sequence-length filtering.

# 🎯 Objectives

The main objectives of this project are:

1. Build an English-to-French Neural Machine Translation system.
2. Implement a Seq2Seq architecture using LSTM.
3. Improve the encoder using Bidirectional LSTM.
4. Implement Additive/Bahdanau Attention.
5. Handle padded sequences correctly.
6. Train the model using teacher forcing.
7. Evaluate translation quality using BLEU.
8. Analyze token-level predictions.
9. Visualize attention weights.
10. Perform qualitative translation analysis.
11. Save the trained model artifacts for future inference.

---

# 📊 Dataset

## OPUS Books — English/French

Dataset:

**Helsinki-NLP/opus_books**

Configuration:

```text
en-fr
```

Source:

Hugging Face Datasets

The dataset provides aligned English and French text pairs.

Example:

```text
English:
The Wanderer

French:
Le grand Meaulnes
```

The notebook loads the dataset using Hugging Face `datasets.load_dataset()` and includes a fallback mechanism that loads the official Parquet file if standard dataset loading fails.

### Original Dataset

```text
Total raw pairs: 127,085
```

### Experiment Dataset

After cleaning and filtering:

```text
Maximum samples: 100,000
Maximum sequence length: 40
```

The notebook removes empty samples, filters sentences longer than the configured maximum length, removes duplicate English–French pairs, shuffles the dataset deterministically, and then selects up to 100,000 examples.

---

# 🔀 Dataset Split

The final dataset is divided into:

| Dataset    |     Samples | Percentage |
| ---------- | ----------: | ---------: |
| Training   |      80,000 |        80% |
| Validation |      10,000 |        10% |
| Test       |      10,000 |        10% |
| **Total**  | **100,000** |   **100%** |

## The split is performed after cleaning and filtering so that the test set remains held out from training.

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

### 1. Lowercasing

English and French sentences are converted to lowercase.

### 2. Character filtering

Unsupported characters are removed while preserving relevant French characters and common punctuation.

### 3. Empty sentence removal

Empty English or French samples are discarded.

### 4. Sequence-length filtering

Only sentence pairs within the configured maximum length are retained.

```python
MAX_LEN = 40
```

### 5. Duplicate removal

Duplicate English–French pairs are removed.

### 6. Start/End tokens

French target sentences are transformed into:

```text
[start] french sentence [end]
```

This allows the decoder to learn when to start and stop generating a translation.

---

# 🔤 Text Vectorization

The project uses TensorFlow/Keras `TextVectorization`.

Configuration:

```python
VOCAB_SIZE = 20_000
MAX_LEN = 40
```

Separate vectorizers are created for:

```text
English → Source Vectorizer
French  → Target Vectorizer
```

The target vocabulary contains special tokens:

```text
[start]
[end]
```

The vectorizers are fitted only on the training data.

---

# 🧠 Model Architecture

## Seq2Seq + BiLSTM + Additive Attention

```text
                    ENGLISH INPUT
                         │
                         ▼
                 Text Vectorization
                         │
                         ▼
                  Embedding (256)
                         │
                         ▼
              Bidirectional LSTM
                  256 units
                         │
              ┌──────────┴──────────┐
              │                     │
       Forward States        Backward States
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
               State Projection
                         │
                         ▼
                  LSTM Decoder
                   512 units
                         │
                         ▼
              Additive Attention
                         │
                         ▼
            Decoder Output + Context
                         │
                         ▼
                    Dropout
                     0.30
                         │
                         ▼
                 Dense + Softmax
                         │
                         ▼
                 FRENCH OUTPUT
```

## The notebook uses a **256-dimensional embedding**, **256-unit Bidirectional LSTM encoder**, **512-unit decoder**, and **0.30 dropout**.

# 🔄 Bidirectional LSTM Encoder

The encoder reads the English sentence in both directions:

```text
Forward:
English → → → → →

Backward:
← ← ← ← ← English
```

This provides the encoder with contextual information from both previous and subsequent tokens.

The encoder generates:

* Encoder sequence outputs
* Forward hidden state
* Forward cell state
* Backward hidden state
* Backward cell state

The forward and backward states are concatenated and projected into the decoder state space.

---

# 🎯 Additive Attention

The project implements a custom **Bahdanau-style Additive Attention** layer.

Conceptually:

```text
score(q, k) = vᵀ tanh(Wq + Uk)
```

where:

* `q` = decoder query
* `k` = encoder representation
* `W` = query projection
* `U` = value projection
* `v` = score projection

The attention layer calculates attention scores between decoder states and encoder outputs and converts them into attention weights using softmax.

The resulting context vector is then combined with the decoder output.

---

# 🛡️ Padding-Aware Attention

A major improvement in this version is explicit padding handling.

Instead of relying on automatic Keras mask propagation, the custom attention layer explicitly masks encoder padding positions.

This was introduced to avoid the previous Keras `BroadcastTo`/mask-shape issue associated with automatic mask propagation through the attention layer.

The model therefore uses:

```python
mask_zero=False
```

for the embeddings and handles encoder padding explicitly inside the custom attention layer.

---

# 👨‍🏫 Teacher Forcing

During training, the decoder uses teacher forcing.

For example:

```text
Decoder Input:
[start] je suis étudiant

Decoder Target:
je suis étudiant [end]
```

More generally:

```text
decoder input  = [start] token1 token2 token3 ...
decoder target = token1 token2 token3 ... [end]
```

This allows the decoder to learn the next-token prediction task efficiently.

---

# 📉 Loss Function

The project uses:

```python
SparseCategoricalCrossentropy
```

with a custom masking mechanism.

Padding tokens are excluded from the loss calculation.

Conceptually:

```text
Loss
 ↓
Calculate token loss
 ↓
Create padding mask
 ↓
Ignore padding positions
 ↓
Average over valid tokens
```

This provides a more meaningful training objective for variable-length sequences.

---

# 📈 Evaluation Metrics

The project does not rely only on token accuracy.

The following evaluation methods are included:

### 1. Masked Token Accuracy

Measures token-level prediction accuracy while ignoring padding tokens.

### 2. BLEU

BLEU is calculated by comparing generated French translations against reference French sentences.

The notebook evaluates up to:

```text
2,000 test examples
```

by default for faster evaluation, while the complete test set can also be evaluated.

### 3. Confusion Matrix

A compact token-level confusion matrix is created using frequent target tokens.

This is intended as a diagnostic visualization rather than a sentence-level translation metric.

### 4. Attention Heatmap

The attention heatmap shows which English source positions receive stronger attention while generating French output tokens.

### 5. Qualitative Translation Comparison

Generated translations are compared directly with reference French sentences.

---

# ⚙️ Training Configuration

Main experiment parameters:

```python
SEED = 42

MAX_SAMPLES = 100000
MAX_LEN = 40
VOCAB_SIZE = 20000

EMBED_DIM = 256

ENCODER_UNITS = 256
DECODER_UNITS = 512

DROPOUT_RATE = 0.30

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 5e-4
```

These settings are defined directly in the notebook.

---

# 🚀 Training Improvements

Compared with the earlier baseline, this version includes several improvements:

* Larger training dataset
* Bidirectional encoder
* Larger embedding dimension
* Larger decoder
* Custom Additive Attention
* Explicit padding handling
* Stronger dropout
* AdamW optimization
* Weight decay
* Gradient clipping
* Adaptive learning-rate reduction
* Early stopping
* Best-model checkpointing
* BLEU evaluation
* Attention visualization

The notebook specifically notes that translation quality should not be judged by validation accuracy alone; BLEU and qualitative translations should also be considered.

---

# 🔮 Inference

After training, the model supports autoregressive decoding.

The decoder generates French tokens one at a time:

```text
[start]
   ↓
French token 1
   ↓
French token 2
   ↓
French token 3
   ↓
...
   ↓
[end]
```

The inference implementation supports:

### Greedy decoding

```python
temperature = 0.0
```

### Temperature sampling

```python
temperature > 0
```

### Repeated-token penalty

Helps reduce undesirable repeated output tokens.

### Early stopping

Generation stops when:

```text
[end]
```

is generated.

---

# 🌡️ Temperature Sampling

For deterministic evaluation:

```python
temperature = 0.0
```

is recommended.

A value such as:

```python
temperature = 0.7
```

can generate alternative translations.

However, sampled output should not be used as the primary BLEU benchmark because sampling introduces stochasticity.

---

# 📊 Visualizations

The notebook includes several visualization components.

## Training Curves

Training and validation:

```text
Loss
Accuracy
```

can be inspected to identify possible overfitting.

## Token Confusion Matrix

Shows common token-level prediction errors.

## Attention Heatmap

Illustrates the alignment between:

```text
English source tokens
        ↕
French generated tokens
```

## Translation Comparison

Example structure:

| English        | Reference French | Generated French |
| -------------- | ---------------- | ---------------- |
| Input sentence | Ground truth     | Model output     |

These visualizations help analyze the model beyond a single numerical metric.

---

# 💾 Model Artifacts

The notebook creates:

```text
models/
visualization/
```

directories for experiment outputs.

The saving section stores model-related artifacts such as:

```text
Model weights
Vocabulary
Configuration
```

These artifacts can be used to reproduce inference without rebuilding the complete training pipeline.

---

# 📁 Recommended GitHub Structure

```text
Seq2Seq_BiLSTM_Attention_OPUS_English_French/
│
├── Seq2Seq_BiLSTM_Attention_OPUS_English_French_100K_Final_Fixed.ipynb
│
├── README.md
├── requirements.txt
│
├── models/
│   ├── best_model.weights.h5
│   ├── source_vocabulary.json
│   ├── target_vocabulary.json
│   └── config.json
│
├── visualization/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── attention_heatmap.png
│
└── LICENSE
```

> The exact filenames inside `models/` should match the artifacts actually generated by your notebook.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Hugging Face Datasets
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* NLTK

The notebook imports TensorFlow, Hugging Face `datasets`, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn and NLTK for the complete pipeline.

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Seq2Seq_BiLSTM_Attention_OPUS_English_French
```

Install dependencies:

```bash
pip install -U datasets huggingface_hub pyarrow tensorflow scikit-learn pandas numpy matplotlib seaborn nltk
```

The notebook uses this package set in its installation section.

---

# ▶️ How to Run

## Option 1 — Google Colab

Upload the notebook to Google Colab and execute the cells sequentially.

Recommended:

```text
Runtime
   ↓
Change runtime type
   ↓
GPU
```

Then run:

```text
1. Install dependencies
2. Load dataset
3. Clean and filter data
4. Split dataset
5. Build vectorizers
6. Build tf.data pipeline
7. Build model
8. Train
9. Evaluate
10. Run inference
11. Calculate BLEU
12. Generate visualizations
13. Save artifacts
```

---

# 🧪 Reproducibility

The experiment uses:

```python
SEED = 42
```

and sets random seeds for:

```text
Python
NumPy
TensorFlow
```

This makes the experiment more reproducible, although exact results can still vary depending on hardware, TensorFlow/Keras versions and execution environment.

---

# ⚠️ Important Notes

### Translation accuracy is not the same as classification accuracy

For machine translation, token-level accuracy alone does not completely describe translation quality.

Therefore this project evaluates:

```text
Masked Token Accuracy
        +
BLEU
        +
Qualitative Translation
        +
Attention Visualization
```

The notebook explicitly avoids assuming a fixed validation accuracy or BLEU target.

### GPU Environment

The notebook records TensorFlow/GPU environment information and notes that native Windows TensorFlow versions ≥2.11 generally do not use NVIDIA CUDA GPUs directly; WSL2/Linux or another supported GPU environment may be preferable for NVIDIA GPU training.

---

# 🔬 Limitations

This project uses a classical recurrent Seq2Seq architecture rather than a Transformer-based NMT architecture.

Potential limitations include:

* LSTM training can be slower than Transformer architectures.
* Long sequences can be difficult for recurrent models.
* Vocabulary size is limited to 20,000 tokens.
* Maximum sequence length is limited to 40 tokens.
* OPUS Books contains literary/book-domain language, so performance may differ on conversational or modern-domain text.
* BLEU does not capture every aspect of translation quality.
* Token accuracy does not necessarily correspond directly to human translation quality.

---

# 🚀 Future Improvements

Possible future extensions include:

* Transformer-based encoder-decoder
* Multi-head attention
* Subword tokenization
* SentencePiece
* Byte Pair Encoding (BPE)
* Larger multilingual datasets
* Beam search
* Length normalization
* Coverage mechanism
* Scheduled sampling
* Label smoothing
* ROUGE / METEOR / chrF evaluation
* Hugging Face Transformers
* T5-based translation
* BART-based translation
* mT5 / mBART multilingual translation
* FastAPI inference API
* Streamlit translation interface
* Docker deployment
* Cloud deployment

---

# 📚 Learning Concepts Demonstrated

This project demonstrates several important Deep Learning and NLP concepts:

```text
Natural Language Processing
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Vocabulary
        ↓
Word Embedding
        ↓
Sequence Modeling
        ↓
LSTM
        ↓
Bidirectional LSTM
        ↓
Seq2Seq
        ↓
Teacher Forcing
        ↓
Attention Mechanism
        ↓
Autoregressive Decoding
        ↓
BLEU Evaluation
        ↓
Model Visualization
```

---

# 👨‍💻 Author

**Md Emon Islam**

Deep Learning | Machine Learning | NLP | Computer Vision | Generative AI

---

# 📄 License

This project can be distributed under the MIT License.

See:

```text
LICENSE
```

for details.

---

# ⭐ Acknowledgement

This project uses the **Helsinki-NLP OPUS Books English–French dataset** through the Hugging Face Datasets ecosystem.

The project is intended for educational, research and portfolio purposes.

---

# ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub and sharing it with other learners interested in:

* Deep Learning
* NLP
* Neural Machine Translation
* LSTM
* Attention Mechanisms
* Sequence-to-Sequence Models
* Transformers
