# Bidirectional Sequence Modeling on IMDb

GitHub-ready NLP project using Hugging Face's `stanfordnlp/imdb` dataset.

## Models
- Bidirectional LSTM (BiLSTM)
- Bidirectional GRU (BiGRU)

## Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve
- AUC
- Training/validation curves
- Model comparison

## Dataset

The project loads:

```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")
```

IMDb contains 25,000 labeled training reviews and 25,000 labeled test reviews, plus an unlabeled split. Labels are `0 = negative` and `1 = positive`.

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/Bidirectional_IMDb_Sequence_Modeling.ipynb
```

## Architecture

```text
Raw Text
   ↓
TextVectorization
   ↓
Embedding
   ↓
Bidirectional LSTM / GRU
   ↓
Dropout
   ↓
Dense
   ↓
Sigmoid
   ↓
Sentiment
```

## Why Bidirectional?

A normal recurrent model processes text in one direction. A bidirectional model processes the sequence from both directions and combines the representations. This can provide useful contextual information for text classification.

## Project structure

```text
bidirectional-imdb-sequence-model/
├── notebook/
│   └── Bidirectional_IMDb_Sequence_Modeling.ipynb
├── models/
├── src/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## GitHub

```bash
git init
git add .
git commit -m "Add Bidirectional IMDb sequence modeling"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bidirectional-imdb-sequence-model.git
git push -u origin main
```

Large `.keras` files should generally be stored with Git LFS, GitHub Releases, or the Hugging Face Model Hub rather than ordinary Git history.

## Dataset citation

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL-HLT 2011.

Dataset: `stanfordnlp/imdb` on Hugging Face. Check its dataset card for the current license/usage terms.
