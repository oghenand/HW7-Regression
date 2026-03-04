"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor, BaseRegressor
from regression.utils import loadDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# (you will probably need to import more things here)

def test_prediction():
	"""
	Test cases for prediction function for LogisticRegressor()
	"""
	X = np.arange(18).reshape((6,3)) / 18 # create initialized array
	X_w_bias = np.hstack([X, np.zeros(shape=(X.shape[0],1))]) # add bias term
	num_feats = X.shape[1] # get in number of features
	num_samples = X.shape[0] # get number of samples 
	logreg = LogisticRegressor(num_feats=num_feats) # initialize logreg object
	y_pred = logreg.make_prediction(X_w_bias) # make prediction

	assert np.array_equal(y_pred, y_pred.astype(bool)), "Predictions must be binary!"
	assert len(y_pred) == num_samples, "Predictions must be same length as number of samples"
	with pytest.raises(ValueError):
		logreg.make_prediction(np.hstack([X_w_bias, np.zeros(shape=(X_w_bias.shape[0],1))]))

def test_loss_function():
	"""
	Test cases for the BCE loss function.
	Function assumes that BCE loss will be used.
	"""
	# use simple true and pred arrays
	y_true_ex = np.array([0,1,0,1,0,0])
	y_pred_ex = np.array([1,0,0,1,0,1])

	# add eps to prevent log errors (numerical stability)
	eps = 1e-8 # epsilon value
	y_true_calc = np.clip(y_true_ex, eps, 1-eps) # np.clip to keep values stablly bounded
	y_pred_calc = np.clip(y_pred_ex, eps, 1-eps)

	# calculate BCE from scratch and compare to logreg-extracted BCE
	bce_true = -np.mean(y_true_calc*np.log(y_pred_calc) + (1-y_true_calc)*np.log(1-y_pred_calc))
	logreg = LogisticRegressor(num_feats=1)
	bce_logreg = logreg.loss_function(y_true_ex, y_pred_ex)
	assert np.abs(bce_true - bce_logreg) < 1e-6, "Calculated and LogisticRegressor-returned BCE unequal!"
	assert bce_logreg >= 0, "BCE loss must be zero at minimum!"

	# check for inproper inputs
	with pytest.raises(ValueError):
		logreg.loss_function(y_true_ex, None) # null/None arrays
	with pytest.raises(ValueError):
		logreg.loss_function(np.array([]), np.array([])) # empty arrays
	with pytest.raises(ValueError):
		logreg.loss_function(y_true_ex+1, y_pred_ex) # non-binary arrays
	with pytest.raises(ValueError):
		logreg.loss_function(np.array([1,0,1,0]), y_pred_ex) # unequal length

def test_gradient():
	# test for inequal size
	# test for gradients being equal to number of weights
	"""
	Test cases for gradient calculation.
	Uses simple arrays to test edge cases and expected results.
	"""
	y_true_ex = np.array([0,1,0,1,0,0]) # example y_true
	X_ex = np.arange(18).reshape((6,3)) / 18 # ex normalized array
	# add bias term
	X_ex_w_bias = np.hstack([X_ex, np.zeros(shape=(X_ex.shape[0],1))])
	num_feats = X_ex.shape[1] # extract number of feats for logreg init.
	logreg = LogisticRegressor(num_feats=num_feats) # init logreg
	gradient = logreg.calculate_gradient(y_true_ex, X_ex_w_bias) # calculate gradient
	
	# gradient should be same length as weights
	assert len(gradient) == len(logreg.W), 'The number of gradients calculated must be equal to number of weights'
	with pytest.raises(ValueError): # raise error if mismatch in X and W shapes
		logreg.make_prediction(np.hstack([X_ex_w_bias, np.zeros(shape=(X_ex_w_bias.shape[0],1))]))

def test_training():
	# load data from dataset
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)

	# now normalize data
	normalizer = StandardScaler()
	X_train = normalizer.fit_transform(X_train)
	X_val = normalizer.transform(X_val)

	# train logistic regression
	logreg = LogisticRegressor(num_feats=X_train.shape[1])
	# save initial and final weights (so pre- and post-training)
	initial_W = logreg.W.copy()
	logreg.train_model(X_train, y_train, X_val, y_val)
	final_W = logreg.W

	# final and initial weights should be different (very highly unlikely that random weights are perfect fit)
	assert np.any(final_W-initial_W != 0), "Initial and Final Weights the same, model did not learn!"
	assert logreg.loss_hist_train[-1] <= logreg.loss_hist_train[0], "Loss should not increase after training!"