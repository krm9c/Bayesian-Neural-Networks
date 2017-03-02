import pymc3 as pm
import theano.tensor as Tensor
import theano
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from   sklearn.datasets import make_moons, make_circles, make_classification
import tflearn
import os,sys
from scipy.stats import mode, chisquare
from sklearn.metrics import confusion_matrix, accuracy_score
import lasagne
from   sklearn.cross_validation      import train_test_split
######################################################################
# For MAC or Unix use this Path
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')
# Set path for the data too
path            = "/Users/krishnanraghavan/Documents/Data-case-study-1"
myRollpath      = "/Users/krishnanraghavan/Documents/InfiniteRollData/"
myRespath       = "/Users/krishnanraghavan/Documents/Results"

#######################################################################
# Now import the required libraries
from Library_Paper_two  import *
from Library_Paper_one  import Collect_samples_Bearing
#######################################################################
# Set up parameters for dimension reduction
I = 100000
draw = 7000
sample = 1000
batch_size = 50
n_hidden = 20
number = 3
Faults = 4
dimension = 10
depth = 3
##################################################################################
# A new try at this code again
def Variational_Inference():
	# Import data
	# 0 -- Artificial Data
	# 1 -- Rolling Element Data
	# 2 --  MNIST Data-set
	That, Yhat = DataImport(0)
	# Lets process the data
	Xunscaled, Tree, Y = Inf_Loop_data(1, That, Yhat)
	Tunscaled, Tree = initialize_calculation(T = Tree, Data = Xunscaled, gsize = dimension, par_train = 1)
	# Standardize the data and split it into two data
	scale = preprocessing.StandardScaler().fit(Tunscaled)
	X = scale.transform(Tunscaled)
	TrainX, TestX, TrainY, TestY = train_test_split(X, Y, test_size = 0.20)
	print "Starting the loop "
	# Call the function corresponding to the model
	def Model():
		ann_input  = theano.shared(TrainX)
		ann_output = theano.shared(TrainY)
		Models	= pm.Model()
		init0 = np.random.randn(X.shape[1], n_hidden)
		init1 = np.random.randn(n_hidden, n_hidden)
		init2 = np.random.randn(n_hidden, 4)
		i = 0

		with Models:
			# First Layer
			weights0= pm.Normal('w0', 0, sd=0.1, shape=(X.shape[1], n_hidden), testval=init0)
			act0 = Tensor.nnet.sigmoid(Tensor.dot(ann_input, weights0))
			# Second
			weights1 = pm.Normal('w1'+str(i)+str(1), 0, sd=1, shape=(n_hidden, n_hidden), testval=init1)
			act1     = Tensor.nnet.sigmoid(Tensor.dot(act0, weights1 ))
			# Third Layer
			weights2 = pm.Normal('w2'+str(i)+str(2), 0, sd=0.1, shape=(n_hidden,4), testval=init2)
			act2 = Tensor.nnet.sigmoid(Tensor.dot(act1, weights2 ))
			# Output
			out = pm.Bernoulli('out', act2, observed=ann_output)

		# Define and create our minibatches
		minibatch_tensors = [ann_input, ann_output]
		minibatch_RV = [out]
		from six.moves import zip
		def create_minibatch(Data):
			rng = np.random.RandomState(0)
			while True:
				idx = rng.randint(len(Data), size=50)
				yield Data[idx]
		minibatches = zip(
		create_minibatch(TrainX),
		create_minibatch(TrainY),
		)
		total_size = TrainY.shape[0]
		# Lets go start the trainer
		print "Trainer Initiated"
		# Trainer
		def f():
			return pm.variational.advi_minibatch(n= I, minibatch_tensors=minibatch_tensors, minibatch_RVs=minibatch_RV, minibatches=minibatches, total_size=total_size, learning_rate=1e-2, model = Models, epsilon=1.0)

		# Trace
		v_params = f()
		trace    =  pm.variational.sample_vp(v_params, model = Models, draws=draw)

		# Get all the results
		ann_input.set_value(X)
		ann_output.set_value(Y)
		# Testing Routines
		with Models:
			ppc = pm.sample_ppc(trace, model=Models, samples=sample)

		# Plotting routines
		pred = ppc['out'].mean(axis = 0)
		f, axarr = plt.subplots(2,2)
		n = 0
		p = 0
		c = { 'r': (1.0, 0.0, 0.0), 'w': (1.0, 1.0, 1.0), 'k': (0.0, 0.0, 0.0), 'm': (0.75, 0, 0.75), 'c': (0.0, 0.75, 0.75), 'g': (0.0, 0.5, 0.0), 'y': (0.75, 0.75, 0), 'b': (0.0, 0.0, 1.0) }
		col = ['r', 'b' , 'c', 'g', 'k']
		print 'Length of colors is', len( col )
		for i in range( Faults ):
			if n > 1:
				n = 0
				p = p+1
			axarr[ p, n ].plot(pred[:,i], color = c[col[ i ]])
			axarr[ p, n ].set_title(i)
			n = n + 1
		# Get the accuracy now and print it.
		ann_input.set_value(TestX)
		ann_output.set_value(TestY)
		with Models:
			ppc = pm.sample_ppc(trace, model=Models, samples=sample)
		pred = ppc['out'].mean(axis = 0) > 0.5
		print(i, 'Accuracy {}%'.format((TestY == pred).mean() *100))
		plt.show()
	Model()
if __name__ == '__main__':
	Variational_Inference()
