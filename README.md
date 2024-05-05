# Multi-Label text classification for Scopus Publications using encoder representation from transformers language model (RoBERTa)

![Alt text](https://github.com/RadchaneepornC/ClassificationScopusPaper/blob/main/image/BertViz.gif)

[credit image](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)


## Motivation
To experiment with how the concept of transfer learning, in terms of using an encoder-based transformer pretrained model (RoBERTa in this case), fine-tuned with the Scopus publication dataset, can improve the text classification performance of the model


## Resource
- **datasets**: Full Raw Scopus Dataset: Resource from 2110531 Data Science and Data Engineering Tools, semester 1/2023, Chulalongkorn University, with the support of the CU Office of Academic Resources (2018 - 2023)

- **Encoder Transformer Model**: [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)

## Statical Metric Using


## Methodology

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P2SevookxKc8DmCGicD5FFsdFXFVevYb?usp=sharing)

### I) Install required packages and import important libaries 

```py

#===============Install required packages =====================#

!pip install transformers[torch] datasets evaluate scipy wandb

#================Import Libaries===============================#
# Data processing
import numpy as np

# Modeling

from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TextClassificationPipeline

# Hugging Face Dataset
from datasets import Dataset

# Model performance evaluation
from sklearn import metrics

#for train, validate spliting
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import tqdm


```
### II) Prepare data for the model

<details><summary>Data Importing & Preprocessing</summary>

</details>
<details><summary>Data Splitting</summary>

</details>
<details><summary>Create Custom dataset</summary>

</details>

</details>
<details><summary>Setup DataLoader</summary>

</details>

### III) Custom model architecture for multiclassification

### IV) Make Baseline
### V) Build training and validation loop

### VI) Testing finetune Model

### VII) Continue finetuning

### VIII) Apply LoRA technique for testing how model efficiency change if changing full finetuning to parameter efficient finetuning

## Result and Error Analysis

![Alt text](image/LossCurve.png)

## Reference
