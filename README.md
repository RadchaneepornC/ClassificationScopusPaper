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

```python
  
#clone dataset storing in git repo
! git clone https://github.com/RadchaneepornC/ClassificationScopusPaper

 #import dataset as DataFrame
import pandas as pd
df_train = pd.read_json("/content/ClassificationScopusPaper/dataset/train_student.json")
print(df_train)

df_test = pd.read_json("/content/ClassificationScopusPaper/dataset/test_student.json")
print(df_test)

#Transpose Data with ```.T``` attribute to swap column and row axis
df_train = df_train.T
df_test = df_test.T

print(df_train.head())
print(df_test.head())

#combline content in the column name Title and Abstract and assign as new column name Context
df_train["Context"]=df_train["Title"]+'.'+df_train["Abstract"]
df_test["Context"]=df_test["Title"]+'.'+df_test["Abstract"]

#drop the old column after combline them
df_train.drop(columns=['Title','Abstract'], inplace = True)
df_test.drop(columns=['Title','Abstract'], inplace = True)

#reset index and drop the old index column
df_train=df_train.reset_index()
df_train.drop(columns=['index'], inplace =True)
df_test.reset_index(inplace = True, drop = True)

#rearrange position of the column
df_train = df_train[["Context","Classes"]]

```

</details>

<details><summary>Label Encoding</summary>

```python

#Initial binarizer named MultiLabelBinarizer from scikit-learn library to encode the multi-label class
mlb = MultiLabelBinarizer(sparse_output=False)

#Encode the classes by fitting the MultiLabelBinarizer on the 'Classes' column and transforming thr classes into a binary matrix, this return encoded matrix
encoder_train = mlb.fit_transform(df_train["Classes"])
encoder_test = mlb.fit_transform(df_test["Classes"])

#Convert encoded matrices to DataFrame
encoder_train = pd.DataFrame(encoder_train, columns = mlb.classes_ )
encoder_test = pd.DataFrame(encoder_test, columns = mlb.classes_ )

#Join Encoded DataFrame with original DataFrame
df_train = df_train.join(encoder_train)
df_test = df_test.join(encoder_test)

#create new column named labels storing the label encode list converted from encode class
df_train['labels'] = df_train[mlb.classes_].values.tolist()
df_test['labels'] = df_test[mlb.classes_].values.tolist()

#drop the old label columns
df_train = df_train.drop(columns = ['Classes','AUTO', 'CE', 'CHE', 'CPE', 'EE', 'IE', 'ME'])
df_test = df_test.drop(columns = ['Classes','AUTO', 'CE', 'CHE', 'CPE', 'EE', 'IE', 'ME'])

#convert 'labels' Column value from Int to Float type
def inttofloat(x):
return list(np.float_(x))
df_train['labels'] = df_train['labels'].apply(lambda x : inttofloat(x))
df_test['labels'] = df_test['labels'].apply(lambda x : inttofloat(x))
```



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
