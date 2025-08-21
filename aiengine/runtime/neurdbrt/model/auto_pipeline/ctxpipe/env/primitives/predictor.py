import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from .primitive import Primitive


class RandomForestClassifierPrim(Primitive):
    def __init__(self):
        super(RandomForestClassifierPrim, self).__init__(name="RandomForestClassifier")
        self.id = 1
        self.hyperparams = []
        self.type = "Classifier"
        self.description = "A random forest classifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)."
        self.accept_type = "c"
        self.model = RandomForestClassifier(random_state=0, n_jobs=os.cpu_count())

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = RandomForestClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class AdaBoostClassifierPrim(Primitive):
    def __init__(self):
        super(AdaBoostClassifierPrim, self).__init__(name="AdaBoostClassifier")
        self.id = 2
        self.hyperparams = []
        self.type = "Classifier"
        self.description = "An AdaBoost classifier. An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. This class implements the algorithm known as AdaBoost-SAMME."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = AdaBoostClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class BaggingClassifierPrim(Primitive):
    def __init__(self):
        super(BaggingClassifierPrim, self).__init__(name="BaggingClassifier")
        self.hyperparams = []
        self.id = 3
        self.type = "Classifier"
        self.description = "A Bagging classifier. A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement, then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = BaggingClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class BernoulliNBClassifierPrim(Primitive):
    def __init__(self):
        super(BernoulliNBClassifierPrim, self).__init__(name="BernoulliNBClassifier")
        self.hyperparams = []
        self.id = 4
        self.type = "Classifier"
        self.description = "Naive Bayes classifier for multivariate Bernoulli models. Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = BernoulliNB()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class DecisionTreeClassifierPrim(Primitive):
    def __init__(self):
        super(DecisionTreeClassifierPrim, self).__init__(name="DecisionTreeClassifier")
        self.hyperparams = []
        self.id = 2
        self.type = "Classifier"
        self.description = "A decision tree classifier."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = DecisionTreeClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class ExtraTreesClassifierPrim(Primitive):
    def __init__(self):
        super(ExtraTreesClassifierPrim, self).__init__(name="ExtraTreesClassifier")
        self.hyperparams = []
        self.id = 6
        self.type = "Classifier"
        self.description = "An extra-trees classifier. This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = ExtraTreesClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class GaussianNBClassifierPrim(Primitive):
    def __init__(self):
        super(GaussianNBClassifierPrim, self).__init__(name="GaussianNBClassifier")
        self.hyperparams = []
        self.id = 7
        self.type = "Classifier"
        self.description = "Gaussian Naive Bayes (GaussianNB). Can perform online updates to model parameters via partial_fit method."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = GaussianNB()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class GaussianProcessClassifierPrim(Primitive):
    def __init__(self):
        super(GaussianProcessClassifierPrim, self).__init__(
            name="GaussianProcessClassifierPrim"
        )
        self.hyperparams = []
        self.id = 8
        self.type = "Classifier"
        self.description = "Gaussian process classification (GPC) based on Laplace approximation. The implementation is based on Algorithm 3.1, 3.2, and 5.1 of Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams. Internally, the Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian. Currently, the implementation is restricted to using the logistic link function. For multi-class classification, several binary one-versus rest classifiers are fitted. Note that this class thus does not implement a true multi-class Laplace approximation."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = GaussianProcessClassifier(n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class GradientBoostingClassifierPrim(Primitive):
    def __init__(self):
        super(GradientBoostingClassifierPrim, self).__init__(
            name="GradientBoostingClassifier"
        )
        self.hyperparams = []
        self.id = 9
        self.type = "Classifier"
        self.description = "Gradient Boosting for classification. GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = GradientBoostingClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class KNeighborsClassifierPrim(Primitive):
    def __init__(self):
        super(KNeighborsClassifierPrim, self).__init__(name="KNeighborsClassifier")
        self.hyperparams = []
        self.id = 3
        self.type = "Classifier"
        self.description = "Classifier implementing the k-nearest neighbors vote."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = KNeighborsClassifier(n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class LinearDiscriminantAnalysisPrim(Primitive):
    def __init__(self):
        super(LinearDiscriminantAnalysisPrim, self).__init__(
            name="LinearDiscriminantAnalysisPrim"
        )
        self.hyperparams = []
        self.id = 11
        self.type = "Classifier"
        self.description = "Linear Discriminant Analysis. A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix. The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = LinearDiscriminantAnalysis()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class LinearSVCPrim(Primitive):
    def __init__(self):
        super(LinearSVCPrim, self).__init__(name="LinearSVC")
        self.hyperparams = []
        self.id = 12
        self.type = "Classifier"
        self.description = "Linear Support Vector Classification. Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = LinearSVC(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class LogisticRegressionPrim(Primitive):
    def __init__(self):
        super(LogisticRegressionPrim, self).__init__(name="LogisticRegression")
        self.hyperparams = []
        self.id = 4
        self.type = "Classifier"
        self.description = """Logistic Regression (aka logit, MaxEnt) classifier.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
        self.accept_type = "c"
        self.model = LogisticRegression(solver="liblinear", random_state=0, n_jobs=5)

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        self.model.fit(train_x, train_y)
        pred_y = self.model.predict(test_x)
        return pred_y

    def predict(self, test_x):
        return self.model.predict_proba(test_x)


class NearestCentroidPrim(Primitive):
    def __init__(self):
        super(NearestCentroidPrim, self).__init__(name="NearestCentroid")
        self.hyperparams = []
        self.id = 14
        self.type = "Classifier"
        self.description = "Nearest centroid classifier. Each class is represented by its centroid, with test samples classified to the class with the nearest centroid."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = NearestCentroid()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class PassiveAggressiveClassifierPrim(Primitive):
    def __init__(self):
        super(PassiveAggressiveClassifierPrim, self).__init__(
            name="PassiveAggressiveClassifier"
        )
        self.hyperparams = []
        self.id = 15
        self.type = "Classifier"
        self.description = "Passive Aggressive Classifier"
        self.hyperparams_run = {"default": True}
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = PassiveAggressiveClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class RidgeClassifierPrim(Primitive):
    def __init__(self):
        super(RidgeClassifierPrim, self).__init__(name="RidgeClassifier")
        self.hyperparams = []
        self.id = 16
        self.type = "Classifier"
        self.description = "Classifier using Ridge regression."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = RidgeClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class RidgeClassifierCVPrim(Primitive):
    def __init__(self):
        super(RidgeClassifierCVPrim, self).__init__(name="RidgeClassifierCV")
        self.hyperparams = []
        self.id = 17
        self.type = "Classifier"
        self.description = "Ridge classifier with built-in cross-validation. By default, it performs Generalized Cross-Validation, which is a form of efficient Leave-One-Out cross-validation."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = RidgeClassifierCV()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class SGDClassifierPrim(Primitive):
    def __init__(self):
        super(SGDClassifierPrim, self).__init__(name="SGDClassifier")
        self.id = 18
        self.hyperparams = []
        self.type = "Classifier"
        self.description = "Linear classifiers (SVM, logistic regression, a.o.) with SGD training. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance. This implementation works with data represented as dense or sparse arrays of floating point values for the features. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = SGDClassifier(random_state=0, n_jobs=os.cpu_count(), loss="log")
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class SVCPrim(Primitive):
    def __init__(self):
        super(SVCPrim, self).__init__(name="SVC")
        self.hyperparams = []
        self.id = 5
        self.type = "Classifier"
        self.description = "C-Support Vector Classification. The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples. The multiclass support is handled according to a one-vs-one scheme. For details on the precise mathematical formulation of the provided kernel functions and how gamma, coef0 and degree affect each other, see the corresponding section in the narrative documentation: Kernel functions."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = SVC(random_state=0, probability=True)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class BalancedRandomForestClassifierPrim(Primitive):
    def __init__(self):
        super(BalancedRandomForestClassifierPrim, self).__init__(
            name="BalancedRandomForestClassifier"
        )
        self.hyperparams = []
        self.id = 20
        self.type = "Classifier"
        self.description = "A balanced random forest classifier. A balanced random forest randomly under-samples each boostrap sample to balance it."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = BalancedRandomForestClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class EasyEnsembleClassifierPrim(Primitive):
    def __init__(self):
        super(EasyEnsembleClassifierPrim, self).__init__(name="EasyEnsembleClassifier")
        self.hyperparams = []
        self.id = 21
        self.type = "Classifier"
        self.description = "Bag of balanced boosted learners also known as EasyEnsemble. This algorithm is known as EasyEnsemble [1]. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = EasyEnsembleClassifier(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class RUSBoostClassifierPrim(Primitive):
    def __init__(self):
        super(RUSBoostClassifierPrim, self).__init__(name="RUSBoostClassifier")
        self.hyperparams = []
        self.id = 22
        self.type = "Classifier"
        self.description = "Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = RUSBoostClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class ARDRegressionPrim(Primitive):
    def __init__(self):
        super(ARDRegressionPrim, self).__init__(name="ARDRegression")
        self.hyperparams = []
        self.id = 23
        self.type = "Regressor"
        self.description = "Bayesian ARD regression. Fit the weights of a regression model, using an ARD prior. The weights of the regression model are assumed to be in Gaussian distributions. Also estimate the parameters lambda (precisions of the distributions of the weights) and alpha (precision of the distribution of the noise). The estimation is done by an iterative procedures (Evidence Maximization)"
        self.accept_type = "c_r"

    def can_accept(self, data):
        return self.can_accept_c(data, "Regression")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = ARDRegression()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class AdaBoostRegressorPrim(Primitive):
    def __init__(self):
        super(AdaBoostRegressorPrim, self).__init__(name="AdaBoostRegressor")
        self.hyperparams = []
        self.id = 24
        self.type = "Regressor"
        self.description = "An AdaBoost regressor. An AdaBoost [1] regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases."
        self.accept_type = "c_r"

    def can_accept(self, data):
        return self.can_accept_c(data, "Regression")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = AdaBoostRegressor(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class BaggingRegressorPrim(Primitive):
    def __init__(self):
        super(BaggingRegressorPrim, self).__init__(name="BaggingRegressor")
        self.hyperparams = []
        self.id = 25
        self.type = "Regressor"
        self.description = "A Bagging regressor. A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4]."
        self.accept_type = "c_r"

    def can_accept(self, data):
        return self.can_accept_c(data, "Regression")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = BaggingRegressor(random_state=0, n_jobs=os.cpu_count())
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class MLPClassifierPrim(Primitive):
    def __init__(self):
        super(MLPClassifierPrim, self).__init__(name="MLPClassifier")
        self.hyperparams = []
        self.id = 4
        self.type = "Classifier"
        self.description = "MLP classifier."
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data, "Classification")

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = MLPClassifier(
            random_state=0,
            # alpha=0.0,
            # solver="sgd",
            # learning_rate="adaptive",
            # learning_rate_init=0.1,
            batch_size=512,
            early_stopping=True,
            # max_iter=3000,
        )
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y
