from sentence_transformers import SentenceTransformer, losses, util, InputExample
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

en_train = pd.read_json('Seminar 2 datasets/data/English dataset/train.jsonl', lines=True)
en_train = en_train[en_train["premise"] != ""]
dupl_train = en_train.value_counts()
dupl_train = dupl_train[dupl_train > 1]
en_train = en_train.drop_duplicates(keep="first")



model = SentenceTransformer("all-MiniLM-L6-v2")

en_train = en_train[:1000]


label_mapping = {"Entailment": 0, "Contradiction": 1}
en_train['label_maped'] = en_train['label'].map(label_mapping)

train_examples = []
for _, row in en_train.iterrows():
    example = InputExample(
        texts=[row["premise"], row["hypothesis"]],
        label=float(row["label_maped"])
    )
    train_examples.append(example)


train_loader = DataLoader(train_examples, batch_size=4, shuffle=True)



# loss function -> ContrastiveLoss for detecting contradictions
train_loss = losses.ContrastiveLoss(model=model)

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./contradictions-model"
)