import pandas as pd
import spacy
import numpy as np

nlp=spacy.load("en_core_web_md") # load sentence tokenzation

input_data=pd.read_csv("ideological_books_corpus.csv", header=None, sep="@", names=['label', 'sentence'])

print(input_data)

mapping = {'Liberal': 1, 'Conservative': 2, 'Neutral': 3}

output = input_data.replace({'label': mapping})

print(output)

output.to_csv("idc_raw.csv", sep='@', index=False)

