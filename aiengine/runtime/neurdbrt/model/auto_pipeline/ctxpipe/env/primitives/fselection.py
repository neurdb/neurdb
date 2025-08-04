from copy import deepcopy
from itertools import compress

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.svm import SVR

from .primitive import Primitive


class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name="VarianceThreshold")
        self.id = 1
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureSelection"
        self.description = """Feature selector that removes all low-variance features.

https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html"""
        self.selector = VarianceThreshold()
        self.accept_type = "c_t"
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector.fit(train_x)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(
            self.selector.transform(train_x), columns=final_cols
        )

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        self.selector.fit(test_x)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return test_data_x
