# Attention + Transformer + BERT + T5 + BART + GPT — AG News

NLP benchmark project using a Hugging Face dataset.

## Models
- BiLSTM + Additive Attention
- Custom Transformer Encoder
- BERT
- T5 (text-to-text classification)
- BART
- GPT-2 (GPT family)

## Dataset
AG News: 4-class news topic classification:
World, Sports, Business, Sci/Tech.

The standard Hugging Face AG News dataset has 120,000 training and 7,600 test examples. See the dataset card and Hugging Face documentation before redistribution or commercial use.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebook/Attention_Transformer_BERT_T5_BART_GPT_AGNews.ipynb
```

For Kaggle/Colab, run the installation cell first and enable GPU.

## Resource note
Running every pretrained model together is computationally expensive. The notebook defaults to a reduced subset. Increase `TRAIN_SAMPLES`, `TEST_SAMPLES`, `EPOCHS`, and model max length when GPU memory/time permits.


## Hugging Face
https://huggingface.co/datasets/ag_news

## Citation
Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level Convolutional Networks for Text Classification." NeurIPS 2015.

## License note
Check the current AG News dataset card and the original corpus terms before redistributing the dataset or using it commercially.
