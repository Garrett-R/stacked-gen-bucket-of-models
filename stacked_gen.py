# -------------------------------------------------------------------------------
# Name:     Stacked Generaliation
# Purpose:  This program implements Stacked Generalization (explained in
#           Wolpert,D.H., 1992).  Instead of using a specified classifier
#           though for 'Level Zero' and 'Level One' generalizatoin, we use
#           Bucket of Models to get the optimal classifier.
#
#           This program also implements Cross Validation to test the
#           performance of the classifier system.
#
# Author:   Garrett Reynolds
#
# -------------------------------------------------------------------------------

from __future__ import print_function

from sklearn import svm
from sklearn.cross_validation import (StratifiedKFold, cross_val_score,
                                      LeaveOneOut)
from math import log10
import numpy as np
# Import two functions from the file extra_clf_tools.py
from extra_clf_tools import find_best_clf, equalize_training_set


def test_stack_gen_perf(data_sets, target, cv_eval=None,
                        clf_list_L0=['svm_rbf'], par_n_folds_L0=4,
                        iterations_L0=1,
                        clf_list_L1=['svm_rbf', 'svm_linear', 'svm_poly'],
                        par_n_folds_L1=20, iterations_L1=4,
                        equalize_training=False, verbose=False):
    """Uses cross-validation to test performance of stacked generalization

    The cross-validation for the evaluation of the performance is cv_eval and
    is a SKLearn CV iterator object.
    For each individual training set, find_best_clf is used to optimize the
    choice of classifier as well as parameters
    (ie we use "Bucket of Models", see Wiki's "Ensemble Learning").

    INPUT:
        >data_sets:   list of arrays.  Each array represents a feature set
                      and should be NxD where N is the number of datapoints
                      in training set and D is the number of features
                      (may be different for each element of list)
        >target:      Nx1 array.
        >cv_eval:     the cross-validation used for evaluated performance;
                      an SKLearn CV iterator
                      (default: 10-fold Stratified K-Fold CV)
        >clf_list_L0: a list of the names of classifiers that should be
                      considered for Level 0 generalization.
                      Possible names: 'all','svm_rbf', 'svm_linear',
                      'svm_poly', 'svm_sigmoid', 'log_reg', 'random_forest'
                      Note: L0 generalization will happen many times, so don't
                      pick something too computationally expensive.
        >par_n_folds_L0: Number of folds for the Stratified K-fold
                         cross-validation used for estimating parameters
                         (default=4). Unlike with cv_eval argument, you don't
                         get flexibility to choose type of cross-validation.
        >iterations_L0: How many times the parameter search grid should be
                        redrawn.  Again, since L0 generalization will be
                        executed many times, it's probably best to leave this
                        at its default value of 1.
        >clf_list_L1, par_n_folds_L1, iterations_L1:
                        Similar to the above arguments, but for Level 1
                        generalization, which only happens once, so this
                        these may be more computationally intensive.
        >equalize_training: If true, it will reduce the training set of each
                            fold such that all classes are represented equally
                            (default=false).

    OUTPUT:
        > results: a dictionary containing the mean accuracy and the score for
                   each class individually."""

    # First turn targets into a nice form (ie counting from 0 up) and
    # remember what the original labels were for output at end.
    class_labels, target = np.unique(target, return_inverse=True)
    num_classes = len(class_labels)

    # check if we should set cv_eval to default
    if cv_eval is None:
        # set the default number of folds
        default_num_folds = 10

        # Quick error check
        smallest_cl_size = np.min(np.bincount(target))
        if smallest_cl_size < default_num_folds:
            print("Warning: the smallest class of your data has just " +
                  str(smallest_cl_size) + " datapoints which is insufficient "
                  "for the default cross-validation (which requires "
                  "at least " + str(default_num_folds) + " datapoints).")
            print("I've reduced the number of folds in the cross-validation "
                  "from " + str(default_num_folds) + " to " +
                  str(smallest_cl_size) + ", so it should be fine.")
            default_num_folds = smallest_cl_size

        cv_eval = StratifiedKFold(target, default_num_folds)

    # Initialize a variable to hold individual class scores
    # (list of empty lists for now)
    class_scores = [[] for x in range(num_classes)]
    # count folds for verbose output
    fold_count = 0

    # Create the Bucket Of Models objects for Level 0 generalization.
    # Note: probability must be set to True (see Wolpert '92)
    BOM_L0 = BucketOfModels(clf_list=clf_list_L0, iterations=iterations_L0,
                            par_n_folds=par_n_folds_L0, probability=True)
    # Create the Bucket Of Models objects for Level 1 generalization.
    BOM_L1 = BucketOfModels(clf_list=clf_list_L1, iterations=iterations_L1,
                            par_n_folds=par_n_folds_L1, probability=False)

    # Now perform cross validation
    for train_indices, test_indices in cv_eval:
        fold_count += 1

        # create training set
        train_data_sets = [
            data_set_i[train_indices, :] for data_set_i in data_sets
        ]
        train_target = target[train_indices]
        # create testing set
        test_data_sets = [
            data_set_i[test_indices, :] for data_set_i in data_sets
        ]
        test_target = target[test_indices]

        # Make training set proportional to classes
        if equalize_training:
            # get the indices of an equalized training set
            eq_train_indices = equalize_training_set(train_target)
            # Now make the equalized training sets
            train_data_sets = [
                train_data_set_i[eq_train_indices, :] for
                train_data_set_i in train_data_sets
            ]
            train_target = train_target[eq_train_indices]

        # ERROR-CHECKING: see if par_n_folds_L0 or par_n_folds_L1 are larger
        # than number of elements in smallest class of train_target
        # and fix them if they are.
        _, tr_targ_sorted = np.unique(train_target, return_inverse=True)
        smallest_cl_size = np.min(np.bincount(tr_targ_sorted))
        # The reason this '-1' is here is because in L0 generalization,
        # we actually leave out one datapoint from training set while
        # converting L0 training set to L1 training set.
        if par_n_folds_L0 > smallest_cl_size-1:
            par_n_folds_L0 = smallest_cl_size-1
            BOM_L0.par_n_folds = par_n_folds_L0
            print("Warning: the parameter par_n_folds_L0 was too big, so I"
                  "reduced it to be: " + str(par_n_folds_L0) +
                  ". No problem though.")
        if par_n_folds_L1 > smallest_cl_size:
            par_n_folds_L1 = smallest_cl_size
            BOM_L1.par_n_folds = par_n_folds_L1
            print("Warning: the parameter par_n_folds_L1 was too big, "
                  "so I reduced it to be: " + str(par_n_folds_L1) +
                  ". No problem though.")

        if verbose:
            print("\nStarting on fold number: " + str(fold_count))

        # make predictions for test set
        predictions = predict_stack_gen(train_data_sets, train_target,
                                        test_data_sets, BOM_L0, BOM_L1,
                                        verbose)

        temp_scores = []

        for i in range(len(test_target)):
            # for the i-th test point, determine what class it belongs to
            temp_class = target[test_indices[i]]
            # store either 0 or 1
            temp_success = int(predictions[i] == test_target[i])
            # into the appropriate class score sublist
            class_scores[temp_class].append(temp_success)
            # and for calculating the mean, we save the value
            temp_scores.append(temp_success)

        if verbose:
            print("The performance for the current " +
                  str(int(len(test_target))) + " holdout datapoints was: " +
                  str(round(np.mean(temp_scores)*100, 1)) + "%\n")

        ####
        # End of Cross Validation Loop
        ####

    # we flatten the class scores (which is a list of lists)
    all_scores = [item for sublist in class_scores for item in sublist]
    accuracy = np.mean(np.array(all_scores))

    # start creating the dictionary of results to be returned
    results = {'accuracy': accuracy}

    for i in range(num_classes):
        mean_for_class_i = np.mean(class_scores[i])
        results.update({class_labels[i]: mean_for_class_i})

    return results

########################
# End of test_stack_gen_perf
########################


def predict_stack_gen(train_data_sets, train_target, test_data_sets,
                      BOM_L0, BOM_L1, verbose=False):
    """Make predictions using stacked generalization on the test_data after
        training with all the training sets.

    INPUT:
        >train_data_sets: list of arrays.  Each array represents a feature set
                          and should be NxD where N is number of datapoints
                          in training set and D is number of features
                          (may be different for each element of list)
        >train_target:  Nx1 array.
        >test_data:  list of arrays.  Each array is MxD, where D can vary for
                     each element of list.
        >BOM_L0:  The BucketOfModels object for Level 0 generalization.
        >BOM_L1:  BucketOfModels object for Level 1 generalization.  This one
                  have parameters that are more computationally intensive since
                  it's only run once.
        >verbose:  Verbose output if set to True (default: False)

    OUTPUT:
        >predictions: Mx1 array of class predictions.  If BOM_L1.probability is
                      True, then you'll get back probabilistic predictions."""

    # Get number of datapoints in training and test sets.
    num_train = train_data_sets[0].shape[0]
    num_test = test_data_sets[0].shape[0]

    # Get number of datasets, which equals the number of generalizers we'll be
    # using in Level 0 generalization
    num_sets = len(train_data_sets)

    # Initialize the L1 training set and L1 testing set
    train_data_L1 = np.zeros((num_train, num_sets))
    test_data_L1 = np.zeros((num_test, num_sets))
    # Set these to -1's so it'll be easier to see if some are never acted on.
    train_data_L1.fill(-1)
    test_data_L1.fill(-1)

    # Go through each set, converting it to L1 datapoints and adding it to the
    # L1 data array
    for current_set in range(num_sets):
        partial_train_data_L1, partial_test_data_L1 = \
            convert_train_and_test_to_L1(
                train_data_sets[current_set], train_target,
                test_data_sets[current_set], BOM_L0, verbose
            )

        train_data_L1[:, current_set] = partial_train_data_L1
        test_data_L1[:, current_set] = partial_test_data_L1

    #####
    # Now we do Level 1 generalization
    ####

    clf_L1 = BOM_L1.train_using_BOM(train_data_L1, train_target)

    if verbose:
        print("The classifier used on the Level 1 test set to get "
              "final predictions was: " + str(clf_L1))

    # Decide whether to give probabilistic classification predictions
    if hasattr(clf_L1, 'probability') and clf_L1.probability is True:
        # (we have two indices because predict_proba strangely
        # gives a 2D array output)
        predictions = clf_L1.predict_proba(test_data_L1)[:, 1]
    else:
        predictions = clf_L1.predict(test_data_L1)

    return predictions


def convert_train_and_test_to_L1(train_data_L0, train_target, test_data_L0,
                                 BOM_L0, verbose=False):
    """This converts one of the feature sets which is already split into
    training and testing to Level 1.

    INPUT:
        >train_data_L0: Training data, NxD array, where N is the number of
                        datapoints in training set and D is the number of
                        features for this feature set.
        >train_target: The targets corresponding to the training datapoints.
                       here should only be two unique elements in this array,
                       since the Bucket of Models it uses is meant for binary
                       classification. Nx1 array.
        >test_data_L0:  Testing data, MxD array, where M is number of
                        datapoints in testing set
        >BOM_L0:     The BucketOfModels object that will be used for Level 0
                     generalization.
        >verbose:     Verbose output.

    OUTPUT:
        >partial_train_data_L1: The training data for Level 1, Nx1 array.
                                This is only one of the features of the Level 1
                                training data corresponding to one of the
                                feature sets, so it will be combined with the
                                results of the other feature sets to make the
                                full Level 1 training set before the
                                final generalization.
        >partial_test_data_L1: Mx1 array."""

    # Reformat the training and test data if it has just 1 feature, since it
    # causes Scikit-Learn's cross_val_score(.) to crash
    if len(train_data_L0.shape) == 1:
        train_data_L0 = np.array([[i] for i in train_data_L0])
    if len(test_data_L0.shape) == 1:
        test_data_L0 = np.array([[i] for i in test_data_L0])

    # Find number of training datapoints and test datapoints.
    num_train_points = train_data_L0.shape[0]
    num_test_points = test_data_L0.shape[0]

    # Initialize the output arrays
    train_data_partial_L1 = np.zeros(num_train_points)
    test_data_partial_L1 = np.zeros(num_test_points)
    # Fill it with -1's to more easily see if there is an error down the line
    train_data_partial_L1.fill(-1)
    test_data_partial_L1.fill(-1)

    #####
    # Now, for each datapoint or training set, we leave it out, train the
    # remaining points, then probabilistically predict the result of the left
    # out point.

    # This looks like Cross-Validation, but we just use it to make partitions
    # of the sets
    temp_cv = LeaveOneOut(num_train_points)

    for temp_data_indices, temp_holdout_point_index in temp_cv:

        # train a temporary classifier on remaining points
        temp_clf = BOM_L0.train_using_BOM(train_data_L0[temp_data_indices],
                                          train_target[temp_data_indices])

        # probabilistic prediction on the holdout point;
        # this gives probability for each of the two classes, but we only save
        # the one for Class 1.  (we have two indices because predict_proba
        # strangely gives a 2D array output)
        train_data_partial_L1[temp_holdout_point_index] = \
            temp_clf.predict_proba(
                train_data_L0[temp_holdout_point_index]
            )[0, 1]

    # train classifier on the full training set to use for converting test set
    # to Level 1
    temp_clf = BOM_L0.train_using_BOM(train_data_L0, train_target)

    if verbose:
        print("The classifier chosen for the converting this set's test data "
              "to Level 1 was: " + str(temp_clf))

    # probabilistic prediction for all points of test set, again only keeping
    # probability for Class 1.
    test_data_partial_L1 = temp_clf.predict_proba(test_data_L0)[:, 1]

    return train_data_partial_L1, test_data_partial_L1


# TODO: OK, so delete it right??
# SUBSUMED by convert_train_and_test_to_L1(.)
# def convert_train_to_L1(train_data_L0, train_target, BOM_L0):
# """This takes the Level 0 training data from one of the models and converts
# it into the Level 1 features.
##
# This function gets called by convert_train_and_test_to_L1.
##
# INPUT:
# >train_data_L0: Training data, NxD array, where N is the number of datapoints in training set
# and D is the number of features.
# >train_target: The targets corresponding to the training datapoints.  There should only
# be two unique elements in this array, since the Bucket of
# Models it uses is meant for binary classification. Nx1 array.
# >BOM_L0:     The BucketOfModels object that will be used for Level 0
# generalization.
##
# OUTPUT:
# >partial_train_data_L1: The training data for Level 1, Nx1 array.  This is only one of
# of the features of the Level 1 training data corresponding to
# one of the feature sets, so it will be combined with the results
# of the other feature sets to make the full Level 1 training
# set before the final generalization."""
##
# Initialize the output array which has same shape as training target array
##    train_data_partial_L1 = np.zeros( len(train_target) )
# Fill it with -1's to more easily see if there is an error down the line
# train_data_partial_L1.fill(-1)
##
##    #####
# Now, for each datapoint, we leave it out, train the remaining points,
# then probabilistically predict the result of that left out point.
##
# This looks like Cross-Validation, but we just use it to make partitions of the sets
##    temp_cv = LeaveOneOut( len(train_target) )
##
# for temp_data_indices, temp_holdout_point_index in temp_cv:
##
# train a temporary classifier on remaining points
##        temp_clf = BOM_L0.train_using_BOM( train_data[temp_data_indices], train_target[temp_data_indices] )
##
# probabilistic prediction on the holdout point;
# this gives probability for each of the two classes, but we only save the one for Class 1.
# (we have two indices because predict_proba strangely gives a 2D array output)
##        train_data_partial_L1[temp_holdout_point_index] = temp_clf.predict_proba( train_data[temp_holdout_point_index] ) [0,1]
##
##
# return train_data_partial_L1


class BucketOfModels:
    """This Class implement Bucket of Models.  You may specify the following
    variables upon creation:

        >clf_list: a list of the names of classifiers that should be considered
            during the Bucket of Models.
            Possible names: 'all', 'svm_rbf', 'svm_linear', 'svm_poly',
                            'svm_sigmoid', 'random_forest'

        >iterations: How many times the parameter grid should be redrawn.
                     (You can just leave it at the default of 2)

        >cv: Cross-validation to be used in determining best parameters
             (if this argument not passed, then StratifiedKFold CV is used with
             number of folds set to par_n_folds)

        >par_n_folds: Number of folds for StratifiedKFold CV. This argument
                      only considered if the cv argument is not passed
                      (or cv=None is passed).

        >probability:  Set to True if you want a probabilistic classifier
                       returned."""

    # We just initialize the object with all these parameters instead of having
    # to pass them every time we use Bucket of Models (this was the motivation
    # for making it an object and not just a method).
    def __init__(self, clf_list=['svm_rbf'], iterations=2, cv=None,
                 par_n_folds=10, probability=False):
        self.clf_list = clf_list
        self.iterations = iterations
        self.cv = cv
        self.par_n_folds = par_n_folds
        self.probability = probability

    def train_using_BOM(self, train_data, train_target):
        """
        RETURNS: Trained classifier (using the SKLearn classifier.
        See scikit-learn.org)"""

        # find the best classifier and parameters for this training set
        clf = find_best_clf(train_data, train_target, clf_list=self.clf_list,
                            iterations=self.iterations, cv=self.cv,
                            par_n_folds=self.par_n_folds,
                            probability=self.probability)

        # train this classifier on the data
        clf.fit(train_data, train_target)

        return clf

    #########
    # End of BucketOfModels class
    ########
