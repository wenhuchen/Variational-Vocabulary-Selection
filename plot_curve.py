import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

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

def draw_uncerntainy_curve():
	x = [100, 200, 500, 1000, 2000, 5000, 10000]
	y_max = [51.2, 67.5, 80.4, 85.1, 87.5, 90.5, 91.4]
	y_mean = [29.6, 33.9, 43.9, 50.81, 67.7, 81.5, 91.4]
	y_min = [25.6, 27, 35.1, 42.4, 56, 74.1, 91.4]


	plt.plot(x, y_mean, '#0099ff', label="Mean Accuracy Curve")
	plt.plot(x, y_min, '#0099ff', label="Lower Accuracy Curve")
	plt.plot(x, y_max, '#0099ff', label="Upper Accuracy Curve")
	#plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
	#plt.plot(x, y_pred, 'b-', label=u'Prediction')
	plt.fill(x + x[::-1], y_min + y_max[::-1], '#ffff4d', alpha=.5, ec='None', label='Accuracy Range')
	plt.legend(loc='lower right')
	plt.xlabel("#vocab")
	plt.ylabel("accuracy")
	plt.savefig("accuracy_curve.eps", dpi=1000, format='eps')
	#plt.show()
	#plt.fill(np.concatenate([x, x[::-1]]),
   	# 		 np.concatenate([y_pred - 1.9600 * sigma,
    #               (y_pred + 1.9600 * sigma)[::-1]]),
    #     			alpha=.5, fc='b', ec='None', label='95% confidence interval')

draw_uncerntainy_curve()

def draw_curve():
    def read(string):
        string = string.strip()
        print string
        result = eval(string) 
        result = [float(_) for _ in result]
        return result
    def savefig(f1, f2, name):
        plt.plot(read(f1[1]), read(f1[0]), 'b-', label="Variational")
        plt.plot(read(f2[1]), read(f2[0]), 'r-', label="Frequency")
        plt.title("{} dataset".format(name))
        plt.xlabel('vocab')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("{}.eps".format(name), format="eps", dpi=200)
        plt.clf()

    f1 = open('/tmp/variational_yelp.txt').readlines()
    f2 = open('/tmp/yelp.txt').readlines()
    savefig(f1, f2, 'yelp')

    f1 = open('/tmp/variational_dbpedia.txt').readlines()
    f2 = open('/tmp/dbpedia.txt').readlines()
    savefig(f1, f2, 'dbpedia')

    f1 = open('/tmp/variational_sogou.txt').readlines()
    f2 = open('/tmp/sogou.txt').readlines()
    savefig(f1, f2, 'sogou')

#draw_curve()
