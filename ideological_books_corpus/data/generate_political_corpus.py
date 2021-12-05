import pandas as pd
import spacy
import numpy as np

nlp=spacy.load("en_core_web_md") # load sentence tokenzation

input_data=pd.read_csv("ideological_books_corpus.csv", header=None, sep="@").values # read testing data

train_data = input_data[input_data.index % 10!=0]
test_data = input_data[input_data.index % 10==0]

train_data.to_csv("train.csv", sep='@')
test_data.to_csv("test.csv", sep='@')

train_data=pd.read_csv("train.csv", header=None, sep="@").values # read training data
test_data=pd.read_csv("test.csv", header=None, sep="@").values # read testing data

test=[]
train=[]
with open("political_corpus_test.txt","w",encoding="utf-8") as f_test:
    with open("political_corpus_train.txt", "w", encoding="utf-8") as f_train:
        with open("political_corpus.txt","w",encoding="utf-8") as f:
            for i in range(len(test_data)):
                if i%1000==0:print(i)
                document=nlp(str(test_data[i][1])) # seperate each sentence in a paragraph
                number=0
                for sent in document.sents:
                    number+=1
                    f.write(str(sent)+"\n") # write new line for each sentence seperated above
                    f_test.write(str(sent)+"\n")
                test.append(number)
                f.write("\n")
                f_test.write("\n")

            for i in range(len(train_data)):
                if i%1000==0:print(i)
                document=nlp(str(train_data[i][1]))
                number=0
                for sent in document.sents:
                    number+=1
                    f.write(str(sent)+"\n")
                    f_train.write(str(sent) + "\n")
                train.append(number)
                f.write("\n")
                f_train.write("\n")


print("max sentence length in test =",np.max(test))
print("avg sentence length in test =",np.average(test))
print()
print("max sentence length in train =",np.max(train))
print("avg sentence length in train =",np.average(train))


