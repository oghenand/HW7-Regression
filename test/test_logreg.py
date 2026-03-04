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
	num_feats = X.shape[1]
	num_samples = X.shape[0]
	logreg = LogisticRegressor(num_feats=num_feats)
	y_pred = logreg.make_prediction(X)

	assert np.array_equal(y_pred, y_pred.astype(bool)), "Predictions must be binary!"
	assert len(y_pred) == num_samples, "Predictions must be same length as number of samples"
	with pytest.rasises(ValueError):
		logreg.make_prediction(np.hstack([X, np.zeros(shape=(X.shape[0],1))]))

def test_loss_function():
	"""
	Test cases for the BCE loss function.
	Function assumes that BCE loss will be used.
	"""
	# use simple true and pred arrays
	y_true_ex = np.array([0,1,0,1,0,0])
	y_pred_ex = np.array([1,0,0,1,0,1])
	# add eps to avoig log errors
	eps = 1e-8
	y_pred_ex = np.clip(y_pred_ex, eps, 1-eps)
	bce_true = -np.mean(y_true_ex*np.log(y_pred_ex) + (1-y_true_ex)*np.log(1-y_pred_ex))
	logreg = LogisticRegressor(num_feats=1)
	bce_logreg = logreg.loss_function(y_true_ex, y_pred_ex)
	assert bce_true == bce_logreg, "Calculated and LogisticRegressor-returned BCE unequal!"
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
	y_true_ex = np.array([0,1,0,1,0,0])
	X_ex = np.arange(18).reshape((6,3)) / 18 # ex normalized array
	num_feats = X_ex.shape[1]
	logreg = LogisticRegressor(num_feats=num_feats)
	gradient = logreg.calculate_gradient(y_true_ex, X_ex)
	
	# gradient should be same length as weights
	assert len(gradient) == len(logreg.W), 'The number of gradients calculated must be equal to number of weights'
	with pytest.rasises(ValueError): # raise error if mismatch in X and W shapes
		logreg.make_prediction(np.hstack([X_ex, np.zeros(shape=(X_ex.shape[0],1))]))

def test_training():
	# load data
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)

	# now normalize data
	normalizer = StandardScaler()
	X_train = normalizer.fit_transform(X_train)
	X_val = normalizer.transform(X_val)

	# train logistic regression
	logreg = LogisticRegressor(num_feats=X_train.shape[1])
	initial_W = logreg.W.copy()
	logreg.train_model(X_train, y_train, X_val, y_val)
	final_W = logreg.W

	# final and initial weights should be different
	assert np.any(final_W-initial_W != 0), "Initial and Final Weights the same, model did not learn!"
	assert logreg.loss_hist_train[-1] <= logreg.loss_hist_train[0], "Loss should not increase after training!"