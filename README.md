# Variational-Vocabulary-Selection
Code for NAACL19 Paper ["How Large a Vocabulary Does Text Classification Need? A Variational Approach to Vocabulary Selection"](https://arxiv.org/abs/1902.10339)

Requirements:
- Tensorflow 1.90
- Pandas
- Sklearn
- Matplotlib

Download the train/val/test data from [Google Drive](https://drive.google.com/open?id=19hAnqpeJRf8UX4_5q93_eMCUiqteMdna), please put the dataset folders in the repo's root directory.

The documents of train.py parameters are listed as follows:
```
1. --model, which model you are using, we provide RNN and CNN models
2. --id, the name of the model you will be saving in the directory, it needs to match during evaluation
3. --variational, a switch, on means using VVD mode
4. --l1, a switch, on means using l1 mode
5. --dataset, choose a dataset from the the above four (ag_news|dbpedia|sogou|yelp_review)
6. --emb_size, setting the embedding dimension
7. --cutoff, the default is None, which means all the words will be remained in the vocabulary, N means the top N frequent words are remained the vocabulary
```

Train word-level CNN Model:
```
python train.py --model=word_cnn --id=word_cnn_cutoff --cutoff 10000 --dataset=ag_news
```

Evaluate Frequency-based CNN Model:
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --cutoff 10000 --compress
```

Train VVD CNN Model:
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --cutoff 10000 --compress --variational
```
Or Start from scratch
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --compress --variational
```
Evaluate VVD CNN Model:
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --cutoff 10000 --compress --variational
```
Or Start from scratch
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --compress --variational
```

We have already provided the pre-trained models in "dataset/architecture_cutoff N_dim_variational/" (like ag_news/word_cnn_cutoff10000_256_variational) folders,
feel free to reload the model for evaluation, an example is shown below:
```
python train.py --model=word_cnn --id=word_cnn_cutoff10000 --dataset=ag_news --cutoff 10000 --compress --variational
```

Tips:
- Note that --cutoff 10000 is an optional argument, it will first cut off the vocabulary to remain first 10K and then perform variational dropout, if you leave it out, the model will start scratch from the huge vocabulary. They are both ending at the same point. The difference only lies in the convergence time.
- For VVD Training, you can stop when the accuracy reaches the maximum, do not wait until it drops too much.
- Tunning the decay parameters will probably harvest better ACC-VOCAB AUC scores.
