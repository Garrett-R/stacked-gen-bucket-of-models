# -----------------------------------------------------------------------------
# Extra Classifier Tools
#
# Author:      Garrett Reynolds
#
# Created:     23/04/2013
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import (LeaveOneOut, StratifiedKFold,
                                      cross_val_score)
from sklearn import preprocessing
from math import log10


def test_feat_perf(data, target,
                   clf_list=['svm_rbf', 'svm_linear', 'svm_poly'],
                   cv_eval=None, par_n_folds=5, iterations=3,
                   equalize_training=False, verbose=False):
    """Uses cross-validation to test performance of classifying a dataset.

    The cross-validation for the evaluation of the performance is cv_eval and
    is a SKLearn CV iterator object.
    For each individual training set, find_best_clf is used to optimize the
    choice of classifier as well as parameters (ie we use "Bucket of Models",
    see Wiki's "Ensemble Learning").

    INPUT:
        >train_data: the training data, a (num_subjects, num_features)-shaped
                     NumPy array
        >train_target: targets, a (num_subjects)-shaped NumPy array, or a list
                       (can also contain strings as labels)
        >clf_list: a list of the names of classifiers that should be
                   considered.  Possible names: 'all','svm_rbf', 'svm_linear',
                   'svm_poly', 'svm_sigmoid', 'log_reg', 'random_forest'
        >cv_eval: the cross-validation used for evaluated performance; an
                  SKLearn CV iterator (default: 10-fold Stratified K-Fold CV)
        >par_n_folds: Number of folds for the Stratified K-fold
                      cross-validation used for estimating parameters
                      (default=5). Uunlike with cv_eval argument, you don't get
                      flexibility to choose type of cross-validation.
        >iterations: How many times the parameter search grid should be
                     redrawn.
        >equalize_training: If true, it will reduce the training set of each
                            fold such that all classes are represented equally
                            (default=false).


    OUTPUT:
        > results: a dictionary containing the mean accuracy and the score for
                   each class individually."""

    # First turn targets into a nice form (ie counting from 0 up) and remember
    # what the original labels were for output at end.
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

    # Initialize a variable to hold individual class scores (list of lists)
    class_scores = [[] for x in range(num_classes)]
    # count folds for verbose output
    fold_count = 0

    # Now perform cross validation
    for train_indices, test_indices in cv_eval:
        fold_count += 1

        # create training set
        train_data = data[train_indices, :]
        train_target = target[train_indices]
        # create testing set
        test_data = data[test_indices, :]
        test_target = target[test_indices]

        # ERROR-CHECKING: see if par_n_folds is larger than number of elements
        # in smallest class of train_target
        _, tr_targ_sorted = np.unique(train_target, return_inverse=True)
        smallest_cl_size = np.min(np.bincount(tr_targ_sorted))
        if par_n_folds > smallest_cl_size:
            par_n_folds = smallest_cl_size
            print("Warning: the parameter par_n_folds was too big, so I "
                  "reduced it to be: " + str(par_n_folds) + ".")

        # Make training set proportional to classes
        if equalize_training:

            # get the indices of an equalized training set
            eq_train_indices = equalize_train_set(train_target)

            # Now make the equalized training sets
            train_data = train_data[eq_train_indices, :]
            train_target = train_target[eq_train_indices]

        # create the CV object for parameter search
        cv_par_est = StratifiedKFold(train_target, par_n_folds)

        if verbose:
            print("\nStarting on fold number: " + str(fold_count))

        ##########
        # These next three commands train on classifier then
        # make prediction on test set
        ##########

        # find the best classifier and parameters for this training set
        best_clf = find_best_clf(train_data, train_target, clf_list,
                                 iterations, cv_par_est, verbose)
        # train this classifier on the data
        best_clf.fit(train_data, train_target)
        # make predictions
        predictions = np.array(best_clf.predict(test_data))

        ##########

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
########################


def find_best_clf(train_data, train_target,
                  clf_list=['svm_rbf', 'svm_linear', 'svm_poly'], iterations=3,
                  cv=None, par_n_folds=10, verbose=False, probability=False):
    """Finds the best classifier with optimal parameters by using an adaptive
    grid search.

    This uses cross-validation to find the best classifier and the best
    parameters by searching on a logarithmic parameter grid, and then searches
    again on smaller logarithmic grid centered at previous best parameter set.

    INPUT:
        >train_data: the training data, a (num_subjects, num_features) shaped
                     NumPy array
        >train_target: targets, a (num_subjects) shaped NumPy array
        >clf_list: a list of the names of classifiers that should be
                   considered.  Possible names: 'all', 'svm_rbf', 'svm_linear',
                   svm_poly', 'svm_sigmoid', 'log_reg', 'random_forest'
        >iterations: How many times the parameter grid should be redrawn.
        >cv: Cross-validation to be used in determining best parameters
             (if this argument not passed, then StratifiedKFold CV is used with
             number of folds set to par_n_folds)
        >par_n_folds: Number of folds for StratifiedKFold CV. This argument
                      only considered if the cv argument is not passed
                      (or cv=None is passed).
        >probability: Set to true if you want a probabilistic classifier
                      returned (only works for SVM's).


    OUTPUT:
        > An SKLearn estimator object representing the best estimator with the
          best parameters."""

    # Check if we should use default value of cv.
    # (Note: don't use the n_folds keyword argument here since it causes
    # problem with the grid's old version of Scikit Learn)
    if cv is None:
        cv = StratifiedKFold(train_target, par_n_folds)

    # Reformat the training data if it has just 1 feature, since it causes
    # cross_val_score(.) to crash
    if len(train_data.shape) == 1:
        train_data = np.array([[i] for i in train_data])

    # initialize the overall best classifier and best score variables
    best_clf = None
    best_score = 0

    # iteration where best classifier was found (this will let the user know if
    # the extra iterations are helping)
    best_iter = 0

    ##########################
    # Check SVM with rbf kernel
    ##########################
    if 'svm_rbf' in clf_list or 'all' in clf_list:

        # SVM with rbf kernel need scaled data, so we scale the data
        scaled_train_data = train_data
        # I've stopped scaling the data, because it finds a classifier
        # appropriate the scaled data, but then outside this
        # function, the data is not necessarily scaled, so you run into issues.
        # I'm not sure how important scaling is anyway...
        # scaled_train_data = preprocessing.scale(train_data)

        # We now set the initial parameters to search on. Note: these represent
        # powers so C_num values of C are logarithmically spaced between
        # 10^(C_min) and 10^(C_max).  I got these particular bounds from:
        # http://scikit-learn.org/0.11/auto_examples/svm/plot_svm_parameters_selection.html
        # (note: C_min and C_max must be floats)
        C_min = -2.
        C_max = 8.
        C_num = 5
        # Set gamma initial parameters
        gamma_min = -5.
        gamma_max = 3.
        gamma_num = 5

        # Initialize this classifier's score to zero
        # (as opposed to overall best score)
        this_clf_score = 0
        this_clf_best = None
        this_best_iter = 0

        for i in range(iterations):

            # set the parameter grid
            C_range = np.logspace(C_min, C_max, C_num)
            gamma_range = np.logspace(gamma_min, gamma_max, gamma_num)

            #create parameter grid
            par_grid = ((C, gamma) for C in C_range for gamma in gamma_range)

            for C, gamma in par_grid:
                temp_clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=probability)
                # Now we test the performance of this temporary classifier, by taking the mean of all the runs of the CV.
                temp_score = cross_val_score(temp_clf, scaled_train_data, train_target, cv=cv).mean()
                # check if we have new best score for this
                if temp_score > this_clf_score:
                    this_clf_score = temp_score
                    this_clf_best = temp_clf
                    this_best_iter = i

            # Update the parameter grid search to be smaller grid centered on best parameters
            # note: (C_max-C_min)/C_num is a slightly smaller than the previous intervals (between exponents since its log spacing)
            C_min, C_max = log10(this_clf_best.C) - (C_max - C_min)/C_num, log10(this_clf_best.C) + (C_max - C_min)/C_num
            gamma_min, gamma_max = log10(this_clf_best.gamma) - (gamma_max - gamma_min)/gamma_num, log10(this_clf_best.gamma) + (gamma_max - gamma_min)/gamma_num

        # Check if this classifier beat our overall best classifier
        if this_clf_score > best_score:
            best_score = this_clf_score
            best_clf = this_clf_best
            best_iter = this_best_iter

    ###########
    # end of SVM with rbf kernel
    ###########

    ##########################
    # Check SVM with linear kernel
    ##########################
    if 'svm_linear' in clf_list or 'all' in clf_list:
        # it seems the linear kernel has no parameters, so we already know this is best parameter set for this clf
        this_clf_best = svm.SVC(kernel='linear', probability=probability)
        this_clf_score = cross_val_score(this_clf_best, train_data, train_target, cv=cv).mean()

        # Check if this classifier beat our overall best classifier
        if this_clf_score > best_score:
            best_score = this_clf_score
            best_clf = this_clf_best
            best_iter = 0  # There's only one *iteration* here, so we store 0.

    ###########
    # end of SVM with linear kernel
    ###########

    ##########################
    # Check SVM with polynomial kernel
    # Note the parameter grid we search on is just has sides of just 3 dots,
    # that's because this is very computationally intensive.
    # Also, this classifier stalled a lot, so I played a_round with the upper limits of the parameters
    # to find the maximum range it can go without stalling.
    ##########################
    if 'svm_poly' in clf_list or 'all' in clf_list:

        # Set degree initial parameters (we'll just take these to be integers)
        degree_min = 0.
        degree_max = 3.
        degree_num = 4
        # Set degree initial parameters
        C_min = -2.
        C_max = 4.
        C_num = 3
        # Set coef0 initial parameters
        coef0_min = -2.
        coef0_max = log10(25000)
        coef0_num = 3
        # Set gamma initial parameters
        gamma_min = -2.
        gamma_max = 2.
        gamma_num = 3

        # Initialize this classifier's score to zero
        this_clf_score = 0
        this_clf_best = None
        this_best_iter = 0

        for i in range(iterations):

            # set the parameter grid
            C_range = np.logspace(C_min, C_max, C_num)
            degree_range = np.linspace(degree_min, degree_max, degree_num)  # note: we linearly space the degree
            coef0_range = np.logspace(coef0_min, coef0_max, coef0_num)
            gamma_range = np.logspace(gamma_min, gamma_max, gamma_num)

            #create parameter grid
            par_grid = ((C, degree, coef0, gamma) for C in C_range
                                                  for degree in degree_range
                                                  for coef0 in coef0_range
                                                  for gamma in gamma_range)

            for C, degree, coef0, gamma in par_grid:
                # I had to add this since when the parameters are too big, for some reason it just stalls.
                if (coef0 > 31000) or (degree >= 3 and coef0 > 2500) or (gamma > 200) or (C > 10**6) or (degree >= 2 and C > 1 and coef0 > 1000) or (C > 1000 and coef0 > 5000):
                    continue
                # test the current classifier and save its score into a temporary variable.
                temp_clf = svm.SVC(kernel='poly', C=C, degree=degree, coef0=coef0, gamma=gamma, probability=probability)
                temp_score = cross_val_score(temp_clf, train_data, train_target, cv=cv).mean()
                # check if we have new best score for this
                if temp_score > this_clf_score:
                    this_clf_score = temp_score
                    this_clf_best = temp_clf
                    this_best_iter = i

            # Update the parameter grid search to be smaller grid centered on best parameters
            C_min, C_max = log10(this_clf_best.C) - (C_max - C_min)/(C_num+1), log10(this_clf_best.C) + (C_max - C_min)/(C_num+1)
            coef0_min, coef0_max = log10(this_clf_best.coef0) - (coef0_max - coef0_min)/(coef0_num+1), log10(this_clf_best.coef0) + (coef0_max - coef0_min)/(coef0_num+1)
            gamma_min, gamma_max = log10(this_clf_best.gamma) - (gamma_max - gamma_min)/(gamma_num+1), log10(this_clf_best.gamma) + (gamma_max - gamma_min)/(gamma_num+1)
            # for degree, we just take the best one (to help reduce computational cost), although we could search for a more accurate degree... (TO DO: try it!)
            degree_min = degree_max = this_clf_best.degree
            degree_num = 1

        # Check if this classifier beat our overall best classifier
        if this_clf_score > best_score:
            best_score = this_clf_score
            best_clf = this_clf_best
            best_iter = this_best_iter

    ###########
    # end of SVM with polynomial kernel
    ###########

    ##########################
    # Check SVM with sigmoid kernel
    ##########################
    if 'svm_sigmoid' in clf_list or 'all' in clf_list:

        # Set degree initial parameters
        C_min = -2.
        C_max = 8.
        C_num = 4
        # Set degree initial parameters
        degree_min = 0.
        degree_max = 4.
        degree_num = 5
        # Set coef0 initial parameters
        coef0_min = -2.
        coef0_max = 4.
        coef0_num = 4

        # Initialize this classifier's score to zero
        this_clf_score = 0
        this_clf_best = None
        this_best_iter = 0

        for i in range(iterations):

            # set the parameter grid
            C_range = np.logspace(C_min, C_max, C_num)
            # linearly space this one (again, this all based off suggested
            # parameter values found on the scikit-learn website)
            degree_range = np.linspace(degree_min, degree_max, degree_num)
            coef0_range = np.logspace(coef0_min, coef0_max, coef0_num)

            # create parameter grid
            par_grid = ((C, degree, coef0)
                        for C in C_range
                        for degree in degree_range
                        for coef0 in coef0_range)

            for C, degree, coef0 in par_grid:
                # test the current classifier and save its score into a temporary variable.
                temp_clf = svm.SVC(kernel='sigmoid', C=C, degree=degree, coef0=coef0, probability=probability)
                temp_score = cross_val_score(temp_clf, train_data, train_target, cv=cv).mean()
                # check if we have new best score for this
                if temp_score > this_clf_score:
                    this_clf_score = temp_score
                    this_clf_best = temp_clf
                    this_best_iter = i

            # Update the parameter grid search to be smaller grid centered on best parameters
            C_min, C_max = log10(this_clf_best.C) - (C_max - C_min)/C_num, log10(this_clf_best.C) + (C_max - C_min)/C_num
            coef0_min, coef0_max = log10(this_clf_best.coef0) - (coef0_max - coef0_min)/coef0_num, log10(this_clf_best.coef0) + (coef0_max - coef0_min)/coef0_num
            # for degree, we just take the best one (to help reduce computational cost), although we could search for a more accurate degree... (TO DO: try it!)
            degree_min = degree_max = this_clf_best.degree
            degree_num = 1

        # Check if this classifier beat our overall best classifier
        if this_clf_score > best_score:
            best_score = this_clf_score
            best_clf = this_clf_best
            best_iter = this_best_iter

    ###########
    # end of SVM with sigmoid kernel
    ###########

    ##########################
    # Check with Logistic "Regression"
    ##########################
    if 'log_reg' in clf_list or 'all' in clf_list:

        # Set degree initial parameters
        C_min = -2.
        C_max = 7.
        C_num = 4
        # Set intercept_scaling initial parameters
        # (this range is a total guesses!)
        intercept_scaling_min = -4.
        intercept_scaling_max = 5.
        intercept_scaling_num = 4

        # Initialize this classifier's score to zero
        this_clf_score = 0
        this_clf_best = None
        this_best_iter = 0

        for i in range(iterations):

            # set the parameter grid
            C_range = np.logspace(C_min, C_max, C_num)
            intercept_scaling_range = np.logspace(intercept_scaling_min, intercept_scaling_max, intercept_scaling_num)

            #create parameter grid
            par_grid = ((C, penalty, dual, fit_intercept, intercept_scaling)
                        for C in C_range
                        for penalty in ['l1', 'l2']
                        for dual in [False, True]
                        for fit_intercept in [False, True]
                        for intercept_scaling in intercept_scaling_range)

            for C, penalty, dual, fit_intercept, intercept_sclaing in par_grid:
                # If fit_intercept is False, then I think intercept_scaling doesn't matter, so just go through it once.
                if fit_intercept is False and intercept_scaling in intercept_scaling_range[1:]:
                    continue
                # penalty='l1' is apparently only supported when dual is False
                if penalty == 'l1' and dual is not False:
                    continue

                # test the current classifier and save its score into a temporary variable.
                temp_clf = LogisticRegression(C=C, penalty=penalty, dual=dual, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
                temp_score = cross_val_score(temp_clf, train_data, train_target, cv=cv).mean()
                # check if we have new best score for this
                if temp_score > this_clf_score:
                    this_clf_score = temp_score
                    this_clf_best = temp_clf
                    this_best_iter = i

            # Update the parameter grid search to be smaller grid centered
            # on best parameters
            C_min, C_max = log10(this_clf_best.C) - (C_max - C_min)/C_num, log10(this_clf_best.C) + (C_max - C_min)/C_num
            intercept_scaling_min, intercept_scaling_max = log10(this_clf_best.intercept_scaling) - (intercept_scaling_max - intercept_scaling_min)/intercept_scaling_num, log10(this_clf_best.intercept_scaling) + (intercept_scaling_max - intercept_scaling_min)/intercept_scaling_num

        # Check if this classifier beat our overall best classifier
        if this_clf_score > best_score:
            best_score = this_clf_score
            best_clf = this_clf_best
            best_iter = this_best_iter

    ###########
    # end of Logistic "Regression"
    ###########

    ###########
    # Test Random Forest Classifier
    ###########
    if 'random_forest' in clf_list or 'all' in clf_list:
        for max_features in [None, "auto", "sqrt", "log2"]:
            temp_clf = RandomForestClassifier(n_estimators=10, max_features=max_features)
            temp_score = cross_val_score(temp_clf, train_data, train_target, cv=cv).mean()

            # Check if this classifier beat our overall best classifier
            if temp_score > best_score:
                best_score = temp_score
                best_clf = temp_clf
                best_iter = 0  # Like linear SVM, only one *iteration* for this classifier
    ###########
    # End of Random Forest Classifier
    ###########

    # make sure a classifier was actually found
    if best_clf is None:
        print("\nERROR: No classifiers were found.  Are you sure that your "
              "classifier list ('" + str(clf_list) + "') contained a valid "
              "classifier (such as 'svm_rbf')?")
        print("Exitting...")
        exit()

    if verbose:
        print("For this training set, the best classifier scored: " +
              str(best_score) + ", was found on iteration number: " +
              str(best_iter+1) + " of adaptive parameter search and "
              "classifier was: " + str(best_clf))

    # Finally return the best classifier.
    return best_clf

    ############################
    # End of find_best_clf(.)
    ############################


def equalize_training_set(train_target):
    """Find indices which will make training set have an equal number of each
    class in it by discarding datapoints from over-represented classes.
    The particular datapoints to  discard are chosen randomly (seeded from
    /dev/urandom, if available, or the clock).

        INPUT:
            >train_target: nx1 array, where n is the number of datapoints in
                           training set.

        OUTPUT:
            >eq_train_indices: list of the indices which represent an equalized
                               training set. m is less than or equal to n,
                               where (n-m) is the number of discarded
                               datapoints."""

    # First turn targets into a nice form (ie counting from 0 up)
    class_labels, target = np.unique(train_target, return_inverse=True)
    num_classes = len(class_labels)

    # find the size of smallest class
    smallest_cl_size = np.min(np.bincount(target))

    # initialize the output inidices
    eq_train_indices = []

    for class_i in range(num_classes):

        # find indices of class i
        temp_indices = np.where(target == class_i)[0]
        # permutate them (BTW, does this really add any benefit?)
        temp_indices = np.random.permutation(temp_indices)
        # add some of these indices to our indices list
        eq_train_indices.extend(list(temp_indices[:smallest_cl_size]))

    return eq_train_indices
