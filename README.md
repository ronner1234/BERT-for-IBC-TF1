# How to Fine-Tune BERT for Classification of Political Speeches?

This is the code and source for the paper [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583) and a fork of the repository [
BERT4doc-Classification](https://github.com/xuyige/BERT4doc-Classification). In the above paper, several experiments to investigate different fine-tuning methods of BERT on text classification task were conducted and provide a general solution for BERT fine-tuning.


## Requirements

For further pre-training, we borrow some code from Google BERT. Thus, we need:

+ tensorflow==1.1x
+ spacy
+ pandas
+ numpy

Note that you need Python 3.7 or earlier for compatibility with tensorflow 1.1x.

For fine-tuning, we borrow some codes from pytorch-pretrained-bert package (now well known as transformers). Thus, we need:

+ torch>=0.4.1,<=1.2.0



## Run the code

### 1) Prepare the data set:

#### Ideological Books Corpus

The source of the data set: [Ideological Books Corpus](https://people.cs.umass.edu/~miyyer/ibc/index.html)

"The Ideological Books Corpus (IBC) consists of 4,062 sentences annotated for political ideology at a sub-sentential 
level as described in our paper. Specifically, it contains 2025 liberal sentences, 1701 conservative sentences, 
and 600 neutral sentences. Each sentence is represented by a parse tree where annotated nodes are associated with a 
label in {liberal, conservative, neutral}." 

To obtain the full dataset, or for any questions / comments about the data, send an email at miyyer@umd.edu.

#### Prepare the Data for the Model

We devided the Ideological Books Corpus into two csv files. One for training the model, the other for testing the model:

```shell
1: train.csv  (80% of data)
2: test.csv   (20% of data)
```

### 2) Download Google BERT-Base:

Download and extract the zip into ``./BERT`` :

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

### 3) Further Pre-Training:

#### Generate Further Pre-Training Corpus

Run the python script to generate :
```shell
python generate_political_corpus.py
```
Output files ``political_corpus.txt, political_corpus_test.txt,`` and `` political_corpus_train.txt`` can be found in directory ``./data``.

#### Create Further Pre-Training Tensorflow File

Create a tensorflow record using the ``political_corpus.txt`` data set for pre-training

```shell
python create_pretraining_data.py --input_file=../data/political_corpus.txt --output_file=../data/tf_political_corpus.tfrecord --vocab_file=../BERT/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5
```
#### Run Pre-Training

Run pretraining using ``tf_political_corpus.tfrecord``

Number of training steps: 100.000\
Save checkpoints after steps: 10.000


```shell
python run_pretraining.py --input_file=../data/tf_political_corpus.tfrecord --output_dir=../BERT/pretraining_output --do_train=True --do_eval=True --bert_config_file=../BERT/bert_config.json --init_checkpoint=../BERT/bert_model.ckpt --train_batch_size=20 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=100000 --num_warmup_steps=10000 --save_checkpoints_steps=10000 --learning_rate=5e-5
```


### 4) Fine-Tuning

#### Convert Tensorflow checkpoint to PyTorch checkpoint

replace checkpoint with (.ckpt-100000)

```shell
python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path ../BERT/pretraining_output/model.ckpt-0 --bert_config_file ../BERT/bert_config.json --pytorch_dump_path ../BERT/pretraining_output/pytorch_model.bin
```

#### Fine-Tuning on downstream tasks

```shell
python run_classifier_single_layer.py --task_name political --do_train --do_eval --do_lower_case --data_dir ../data/ --vocab_file ../BERT/vocab.txt --bert_config_file ../BERT/bert_config.json --init_checkpoint ../BERT/pretraining_output/pytorch_model.bin --max_seq_length 512 --train_batch_size 20 --learning_rate 1e-5 --num_train_epochs 8.0 --output_dir ../data/output/ --seed 42 --layers 11 10 --trunc_medium -1
```

where ``num_train_epochs`` can be 3.0, 4.0, or 6.0.

``layers`` indicates list of layers which will be taken as feature for classification.
-2 means use pooled output, -1 means concat all layer, the command above means concat
layer-10 and layer-11 (last two layers).

``trunc_medium`` indicates dealing with long texts. -2 means head-only, -1 means tail-only,
0 means head-half + tail-half (e.g.: head256+tail256),
other natural number k means head-k + tail-rest (e.g.: head-k + tail-(512-k)).

There also other arguments for fine-tuning:

``pooling_type`` indicates which feature will be used for classification. `mean` means
mean-pooling for hidden state of the whole sequence, `max` means max-pooling, default means
taking hidden state of `[CLS]` token as features.

``layer_learning_rate`` and ``layer_learning_rate_decay`` in ``run_classifier_discriminative.py``
indicates layer-wise decreasing layer rate (See Section 5.3.4).

