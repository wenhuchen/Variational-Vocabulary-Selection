from data_utils import *
import pickle
from collections import Counter
import math

if False:
	dataset = "sogou_news"
	#dataset = "yelp_review"

	input_vocab = 'vocab/{}_word_dict_cutoff10000.pickle'.format(dataset)
	train_file = '{}_csv/train.csv'.format(dataset)

	word_dict = pickle.load(open(input_vocab))
	vocab_size = len(word_dict)

	xs, ys = build_word_dataset(dataset, "train", word_dict, 100, tokenize=True)

	examples = len(xs)
	print("totally {} words".format(examples))
	tf_idf_dict = {}
	for d in xs:
		local_counter = Counter(d)
		for word, times in local_counter.iteritems():
			if word not in tf_idf_dict:
				tf_idf_dict[word] = {'tf':times, "idf":1}
			else:
				tf_idf_dict[word]['tf'] += times
				tf_idf_dict[word]['idf'] += 1

	for k in tf_idf_dict:
		tf_idf_dict[k]['idf'] = examples / tf_idf_dict[k]['idf']
		tf_idf_dict[k]['score'] = tf_idf_dict[k]['tf'] * math.log(tf_idf_dict[k]['idf'])

	final_scores = [tf_idf_dict[i]['score'] if i in tf_idf_dict else 0 for i in range(vocab_size)]

	with open('vocab/{}_tf_idf.pickle'.format(dataset), 'w') as f:
		pickle.dump(tf_idf_dict, f)
else:
	dataset = 'snips'
	input_vocab = 'SLU-vocab/{}-data-10000_dict.pickle'.format(dataset)
	train_file = 'SLU-data/{}/train/seq.in'.format(dataset)
	
	word_dict = pickle.load(open(input_vocab))
	xs = open(train_file).readlines()
	vocab_size = len(word_dict['vocab'])
	examples = len(xs)

	print("totally {} words".format(examples))

	tf_idf_dict = {}
	for d in xs:
		d = [word_dict['vocab'][_] if _ in word_dict['vocab'] else 1 for _ in d.strip().split()]
		local_counter = Counter(d)
		for word, times in local_counter.iteritems():
			if word not in tf_idf_dict:
				tf_idf_dict[word] = {'tf':times, "idf":1}
			else:
				tf_idf_dict[word]['tf'] += times
				tf_idf_dict[word]['idf'] += 1

	for k in tf_idf_dict:
		tf_idf_dict[k]['idf'] = examples / tf_idf_dict[k]['idf']
		tf_idf_dict[k]['score'] = tf_idf_dict[k]['tf'] * math.log(tf_idf_dict[k]['idf'])

	final_scores = [tf_idf_dict[i]['score'] if i in tf_idf_dict else 0 for i in range(vocab_size)]
	
	with open('vocab/{}_tf_idf.pickle'.format(dataset), 'w') as f:
		pickle.dump(tf_idf_dict, f)