Stacked Generalization and Bucket of Models
==========================================

This is an implementation of Stacked Generalization, as explained in the paper [Stacked Generalization, 1992, by D.H. Wolpert](http://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf). 

It extends Stacked Generalization by also offering [Bucket of Models](https://en.wikipedia.org/wiki/Ensemble_learning#Bucket_of_models) for the selection of the classifiers and parameters.  The classifiers and parameters are choosing by using cross-validation on the training set.

Usage
=====

There's an included file showing an exmaple usage.


Issues
======

 - Using cross-validation to find optimal parameters is very computationally costly.  It would be better to rewrite this algorithm in C++.

License
=======

GNU GPL version 2 or later

