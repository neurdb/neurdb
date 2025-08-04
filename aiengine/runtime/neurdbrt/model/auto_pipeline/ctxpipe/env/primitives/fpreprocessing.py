from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .primitive import Primitive


def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x


class MinMaxScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(MinMaxScalerPrim, self).__init__(name="MinMaxScaler")
        self.id = 1
        self.gid = 9
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

The transformation is given by:

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
where min, max = feature_range.

This transformation is often used as an alternative to zero mean, unit variance scaling.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html"""
        self.scaler = MinMaxScaler()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        if len(num_test_x.columns) == 0:
            return test_x

        self.scaler.fit(test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class MaxAbsScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(MaxAbsScalerPrim, self).__init__(name="MaxAbsScaler")
        self.id = 2
        self.gid = 10
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Scale each feature by its maximum absolute value.

This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html"""
        self.scaler = MaxAbsScaler()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class RobustScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(RobustScalerPrim, self).__init__(name="RobustScaler")
        self.id = 3
        self.gid = 11
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html"""
        self.scaler = RobustScaler()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class StandardScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(StandardScalerPrim, self).__init__(name="StandardScaler")
        self.id = 4
        self.gid = 12
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"""
        self.scaler = StandardScaler()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class QuantileTransformerPrim(Primitive):
    def __init__(self, random_state=0):
        super(QuantileTransformerPrim, self).__init__(name="QuantileTransformer")
        self.id = 5
        self.gid = 13
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Transform features using quantiles information.

This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme.

The transformation is applied on each feature independently. First an estimate of the cumulative distribution function of a feature is used to map the original values to a uniform distribution. The obtained values are then mapped to the desired output distribution using the associated quantile function. Features values of new/unseen data that fall below or above the fitted range will be mapped to the bounds of the output distribution. Note that this transform is non-linear. It may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html"""
        self.scaler = QuantileTransformer()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class PowerTransformerPrim(Primitive):
    def __init__(self, random_state=0):
        super(PowerTransformerPrim, self).__init__(name="PowerTransformer")
        self.id = 6
        self.gid = 14
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Apply a power transform featurewise to make data more Gaussian-like.

Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html"""
        self.scaler = PowerTransformer()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class NormalizerPrim(Primitive):
    def __init__(self, random_state=0):
        super(NormalizerPrim, self).__init__(name="Normalizer")
        self.id = 7
        self.gid = 15
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = """Normalize samples individually to unit norm.

Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1, l2 or inf) equals one.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html"""
        self.scaler = Normalizer()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x


class KBinsDiscretizerOrdinalPrim(Primitive):
    def __init__(self, random_state=0):
        super(KBinsDiscretizerOrdinalPrim, self).__init__(
            name="KBinsDiscretizerOrdinal"
        )
        self.id = 8
        self.gid = 16
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = "Bin continuous data into intervals. Ordinal."
        self.hyperparams_run = {"default": True}
        self.preprocess = None
        self.accept_type = "c_t_kbins"
        self.need_y = False

    def can_accept(self, data):
        if not self.can_accept_c(data):
            return False
        return True

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
        self.scaler = ColumnTransformer(
            [("discrit", KBinsDiscretizer(encode="ordinal"), list(num_train_x.columns))]
        )
        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(
                self.scaler.transform(num_train_x), columns=list(num_train_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_test_x, num_test_x = catch_num(test_x)
        self.scaler = ColumnTransformer(
            [("discrit", KBinsDiscretizer(encode="ordinal"), list(num_test_x.columns))]
        )
        self.scaler.fit(num_test_x)

        num_test_x = (
            pd.DataFrame(
                self.scaler.transform(num_test_x), columns=list(num_test_x.columns)
            )
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return test_data_x
