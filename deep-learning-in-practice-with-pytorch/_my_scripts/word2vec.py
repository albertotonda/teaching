# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:28:06 2024

@author: Alberto
"""

import torch
import datasets

print("Loading dataset (might need to download data)...")
dataset = datasets.load_dataset('tweets_hate_speech_detection')

# For simplicity let's remove alphanumeric but keep @, #
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


ss = SnowballStemmer('english')
sw = stopwords.words('english')

def split_tokens(row):                             # STEP
    row['all_tokens'] = [ss.stem(i) for i in       # 5
                     re.split(r" +",               # 3
                     re.sub(r"[^a-z@# ]", "",      # 2
                            row['tweet'].lower())) # 1
                     if (i not in sw) and len(i)]  # 4
    return row

# Determine vocabulary so we can create mapping
dataset = dataset.map(split_tokens)