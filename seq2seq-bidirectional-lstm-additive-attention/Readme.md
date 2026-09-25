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

