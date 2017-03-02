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
Faults = 4
dimension = 10
function_choose = "sigmoid"
I = 50000
par = 50
draw = 7000
number = 3
sample = 500
n_hidden = 10
depth = 3
batch_size = 50

###################################################################################
# Create a infinite Loop of Data-stream
def DataImport_Rolling(path, num):
	sheet    = 'Test';
	f        = 'IR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));
	f        = 'OR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));
	f        = 'NL'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'normal';
	f        = 'Normal_'+str(num)+'.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));
	T = np.concatenate((Temp_Norm, Temp_NL, Temp_IR, Temp_OR))
	Y = np.concatenate(( np.zeros(Temp_Norm.shape[0]), (np.zeros(Temp_NL.shape[0])+1), np.zeros(Temp_IR.shape[0])+2, np.zeros(Temp_OR.shape[0])+3 ))
	return T, Y

###################################################################################
# Artificial Dataset import
def DataImport_Artificial(n_sam, n_fea, n_inf ):

	X,y = make_classification(n_samples = n_sam, n_features = n_fea, n_informative = n_inf, n_redundant = (n_fea-n_inf), n_classes = Faults, n_clusters_per_class = 1, class_sep = 2, hypercube = True, shuffle = True, random_state = 9000)

	index_1 = [i for i,v in enumerate(y) if v == 0 ]
	index_2 = [i for i,v in enumerate(y) if v == 1 ]
	index_3 = [i for i,v in enumerate(y) if v == 2 ]
	index_4 = [i for i,v in enumerate(y) if v == 3 ]

	Data_class_1 = X[index_1,:]
	L1 = y[index_1];
	L2 = y[index_2];
	L3 = y[index_3];
	L4 = y[index_4];

	Data_class_2 = X[index_2,:]
	Data_class_3 = X[index_3,:]
	Data_class_4 = X[index_4,:]

	T = np.concatenate((Data_class_1, Data_class_2, Data_class_3, Data_class_4))
	Y = np.concatenate((L1, L2, L3, L4))

	return T, Y

###################################################################################
# Global Import of Data
def Inf_Loop_data(par, That, Yhat):
	if par ==1:
		Ref, Tree = initialize_calculation(T = None, Data = That, gsize = dimension, par_rbf = par, K_function = function_choose, par_train = 0)
		return ( That + 0 * np.random.normal( 0, 0.1, (That.shape[0], That.shape[1]) ) ), Tree, tflearn.data_utils.to_categorical(Yhat, Faults)
	else:
		return (That + 0 * np.random.normal(0, 0.1, (That.shape[0], That.shape[1])) ), tflearn.data_utils.to_categorical(Yhat, Faults)

###################################################################
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data
    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

########################################################################################
def DataImport(num):
	# Import Artificial Data-set
	if num ==0:
		That, Yhat = DataImport_Artificial(n_sam = 5000, n_fea=100, n_inf=6)
	elif num ==1:
		#  Import Rolling Element Data
		That, Yhat = DataImport_Rolling(path, 3)
	elif num ==2:
		X_train, y_train, X_val, y_val, X_test, y_test= load_dataset()
		That = np.concatenate((X_train.reshape(-1, 784), X_test.reshape(-1, 784), X_val.reshape(-1, 784)))
		Yhat = np.concatenate((y_train, y_test, y_val))
	return That, Yhat

########################################################################################
# A new try at this code again
def Variational_Inference():
	# Import data
	# 0 -- Artificial Data
	# 1 -- Rolling Element Data
	# 2 --  MNIST Data-set
	That, Yhat = DataImport(0)

	# Lets process the data
	Xunscaled, Tree, Y = Inf_Loop_data(1, That, Yhat)
	Tunscaled, Tree = initialize_calculation(T = Tree, Data = Xunscaled, gsize = dimension, par_rbf = par, K_function = function_choose, par_train = 1)

	# Standardize the data and split it into two data
	scale = preprocessing.StandardScaler().fit(Tunscaled)
	X = scale.transform(Tunscaled)
	TrainX, TestX, TrainY, TestY = train_test_split(X, Y, test_size = 0.40)
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
			weights2 = pm.Normal('w2'+str(i)+str(2), 0, sd=1, shape=(n_hidden,4), testval=init2)
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
			print "Iteration", '(', p, ', ', n, ',', i, ')'
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
