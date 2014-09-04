
from stacked_gen import BucketOfModels, predict_stack_gen, test_stack_gen_perf
from sklearn import datasets
import numpy as np


# Stacked generalization is useful for when we have multiple sets of data
# describing different aspects of the thing we're predicting.  For example,
# in classifying people, you may have a writing sample, voice sample, data
# on age, gender, etc.  Stacked Generalization is an ensemble learning method
# that attempts to decide which of these datasets is the best predictor.

# to test, we're going to split up the digits dataset three times and pretend
# that these datasets are referring to totally different physical measurements
# (in practice, if you had a single dataset, there's no reason to use Stacked
# Generalization)
digits = datasets.load_digits()

# the number of data points to consider for this example.
num_datapoints = 200

data_sets = [digits.data[:num_datapoints, :20],
             digits.data[:num_datapoints, 20:40],
             digits.data[:num_datapoints, 40:64]]
target = digits.target[:num_datapoints]


results = test_stack_gen_perf(data_sets, target, cv_eval=None,
                              clf_list_L0=['svm_linear'], par_n_folds_L0=3,
                              iterations_L0=1,
                              clf_list_L1=['svm_rbf', 'svm_linear'],
                              par_n_folds_L1=3, iterations_L1=3,
                              equalize_training=True, verbose=True)

print(results)
