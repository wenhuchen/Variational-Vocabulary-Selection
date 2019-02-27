import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from scipy.interpolate import interp1d
from data_utils import *

def integral(y, x):
	area = 0
	for xi, xj, yi, yj in zip(x[:-1], x[1:], y[:-1], y[1:]):
		area += (yi + yj) / 2 * abs(xj - xi)
	return area

def preprocess(x):
	#x = [math.log10(_) for _ in x]
	x = [_/float(max(x)) for _ in x]
	return x

def func(y, x, y_int):
	for y1, y2, x1, x2 in zip(y[:-1], y[1:], x[:-1], x[1:]):
		if y_int == y1:
			return x1
		elif y_int == y2:
			return x2
		elif y_int > y1 and y_int < y2:
			x_int = (y_int - y1) / (y2 - y1) * (x2 - x1) + x1 
			return x_int

def draw_curve():
	x = [1, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
	y_mean = [13.3, 29.6, 33.9, 43.8, 50.81, 67.7, 75.6, 81.5, 91.4, 95.6]
	plt.plot(x, y_mean, 'black')
	#plt.fill(x + x[::-1], y_mean + [95.6] * len(y_min), '#f4df42', alpha=.5, ec='None')
	plt.fill(x + x[::-1], [0] * len(y_mean) + y_mean[::-1], '#0099ff', alpha=.5, ec='None')
	plt.xlabel("Vocab")
	plt.xscale('log')
	plt.xlim(left=0)
	plt.ylim(ymin=0)
	plt.ylabel("Accuracy")
	#plt.show()
	plt.savefig("metrics.eps", dpi=1000, format='eps')

#draw_curve()

def draw_uncerntainy_curve():
	x = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
	y_max = [13.3, 51.2, 67.5, 80.4, 85.1, 87.5, 90.5, 91.4]
	y_mean = [13.3, 29.6, 33.9, 43.9, 50.81, 67.7, 81.5, 91.4]
	y_min = [13.3, 25.6, 27, 35.1, 42.4, 56, 74.1, 91.4]


	plt.plot(x, y_mean, 'black', label="Mean Accuracy Curve")
	plt.plot(x, y_min, 'black', label="Lower Accuracy Curve")
	plt.plot(x, y_max, 'black', label="Upper Accuracy Curve")
	#plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
	#plt.plot(x, y_pred, 'b-', label=u'Prediction')
	plt.fill(x + x[::-1], y_min + y_max[::-1], '#0099ff', alpha=.5, ec='None', label='Accuracy Range')
	plt.legend(loc='lower right', prop={'size':14})
	plt.xlim(left=0)
	plt.xlabel("Vocab")
	plt.ylabel("Accuracy")
	plt.savefig("accuracy_curve.eps", dpi=1000, format='eps')
	#plt.show()
	#plt.fill(np.concatenate([x, x[::-1]]),
   	# 		 np.concatenate([y_pred - 1.9600 * sigma,
    #               (y_pred + 1.9600 * sigma)[::-1]]),
    #     			alpha=.5, fc='b', ec='None', label='95% confidence interval')

#draw_uncerntainy_curve()


def draw_SLU_uncerntainy_curve():
	x = [0, 7, 27, 77, 100, 1778, 5134, 10000]
	x = [str(_) for _ in x]
	y_max = [13.3, 48.8, 81.3, 92.0, 94.0, 95.3, 95.8, 96.1]
	y_mean = [13.3, 33.4, 54.3, 77.4, 88.9, 93.5, 94.2, 96.1]
	y_min = [13.3, 14.2, 33.2, 46.8, 72.8, 88.4, 92.3, 96.1]


	plt.plot(x, y_mean, color='black', label="Mean Accuracy Curve")
	plt.plot(x, y_min, color='black', label="Lower Accuracy Curve")
	plt.plot(x, y_max, color='black', label="Upper Accuracy Curve")
	#plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
	#plt.plot(x, y_pred, 'b-', label=u'Prediction')
	plt.fill(x + x[::-1], y_min + y_max[::-1], color='#0099ff', alpha=.5, ec='None', label='Accuracy Range')
	plt.xlim(left=0)
	plt.ylim(bottom=0)
	plt.legend(loc='lower right', prop={'size':14})
	plt.xlabel("Vocab")
	plt.ylabel("Accuracy")
	plt.savefig("accuracy_curve.eps", dpi=1000, format='eps')
	#plt.show()
	#plt.fill(np.concatenate([x, x[::-1]]),
   	# 		 np.concatenate([y_pred - 1.9600 * sigma,
    #               (y_pred + 1.9600 * sigma)[::-1]]),
    #     			alpha=.5, fc='b', ec='None', label='95% confidence interval')

#draw_SLU_uncerntainy_curve()
def read(string, use_str=False):
    string = string.strip()
    result = eval(string) 
    if use_str:
    	result = [str(_) for _ in result]
    else:
    	result = [float(_) for _ in result]
    return result

def draw_curve():
    def savefig(f1, f2, f3, name):
        x, y = enhanced(read(f1[1]), read(f1[0]))
        plt.plot(x, y, 'y*', label="Frequency")
        x, y = enhanced(read(f2[1]), read(f2[0]))
        plt.plot(x, y, 'b--', label="TF-IDF")
        x, y = enhanced(read(f3[1]), read(f3[0]))
        plt.plot(x, y, 'r', label="Variational")
        #plt.title("{} dataset".format(name))
        plt.xlabel('Vocab')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.xscale('log')
        #plt.xlim(left=0.001)
        #plt.show()
        plt.savefig("{}.eps".format(name), format="eps", dpi=1000)
        plt.clf()
	plt.rcParams.update({'font.size': 14})
    file = 'results/ag_news'
    f1 = open(file + ".txt").readlines()
    f2 = open(file + "_tf_idf.txt").readlines()
    f3 = open(file + "_var.txt").readlines()
    savefig(f1, f2, f3, 'images/ag_news')
    
    file = 'results/dbpedia'
    f1 = open(file + ".txt").readlines()
    f2 = open(file + "_tf_idf.txt").readlines()
    f3 = open(file + "_var.txt").readlines()
    savefig(f1, f2, f3, 'images/dbpedia')

    file = 'results/yelp_review'
    f1 = open(file + ".txt").readlines()
    f2 = open(file + "_tf_idf.txt").readlines()
    f3 = open(file + "_var.txt").readlines()
    savefig(f1, f2, f3, 'images/yelp_review')

#draw_curve()

def compute_score():
	from data_utils import *

	f = 'results/ag_news.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 61673)
	print CR(y, x)

	f = 'results/ag_news_tf_idf.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 61673)
	print CR(y, x)
	
	f = 'results/ag_news_var.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 61673)
	print CR(y, x)
	print()
	
	f = 'results/dbpedia.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 563355)
	print CR(y, x)

	f = 'results/dbpedia_tf_idf.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 563355)
	print CR(y, x)

	f = 'results/dbpedia_var.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 563355)
	print CR(y, x)	
	print()
	
	f = 'results/yelp_review.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 252712)
	print CR(y, x)
	
	f = 'results/yelp_review_tf_idf.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 252712)
	print CR(y, x)
	
	f = 'results/yelp_review_var.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 252712)
	print CR(y, x)
	print()

	f = 'results/sogou_news.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 254495)
	print CR(y, x)
	
	f = 'results/sogou_news_tf_idf.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 254495)
	print CR(y, x)
	
	f = 'results/snli.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 42391)
	print CR(y, x)

	f = 'results/snli_var.txt'
	y = read(open(f).readlines()[0])
	x = read(open(f).readlines()[1])
	print ROC(y, x, 42391)
	print CR(y, x)

compute_score()
