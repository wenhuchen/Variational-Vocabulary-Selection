# Variational-Vocabulary-Selection
Code for NAACL Paper "How Large a Vocabulary Does Text Classification Need? A Variational Approach to Vocabulary Selection", the paper will be coming soon.

Requirements:
- Tensorflow 1.90
- Pandas
- Sklearn
- Matplotlib

Download the data from [Google Drive](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M), please put these four datasets in the repo's root directory.
- AG-News Dataset
- DBPedia Dataset
- Sougou Dataset
- Yelp-Review Dataset


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

Evaluate VVD CNN Model:
```
python train.py --model=word_cnn --id=word_cnn_cutoff --dataset=ag_news --cutoff 10000 --compress --variational
```

We have already provided the pre-trained models in "dataset/architecture_cutoff N_dim_variational/" (like ag_news/word_cnn_cutoff10000_256_variational) folders,
feel free to reload the model for evaluation, an example is shown below:
```
python train.py --model=word_cnn --id=word_cnn_cutoff10000 --dataset=ag_news --cutoff 10000 --compress --variational
```

Tips:
- For VVD Training, you can stop when the accuracy reaches the maximum, do not wait until it drops too much.
- Tunning the decay parameters will probably harvest better ACC-VOCAB AUC scores.
