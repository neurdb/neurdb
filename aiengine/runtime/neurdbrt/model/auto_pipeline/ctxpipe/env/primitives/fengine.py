import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import (
    NMF,
    PCA,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    LatentDirichletAllocation,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from .primitive import Primitive


class PolynomialFeaturesPrim(Primitive):
    def __init__(self, random_state=0):
        super(PolynomialFeaturesPrim, self).__init__(name="PolynomialFeatures")
        self.id = 1
        self.gid = 17
        self.hyperparams = []
        self.type = "FeatureEngine"
        self.description = """Generate polynomial and interaction features.

Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html"""
        self.scaler = PolynomialFeatures(include_bias=False)
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.scaler.fit(train_x)

        train_data_x = self.scaler.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x)
        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        self.scaler.fit(test_x)

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return test_data_x


class InteractionFeaturesPrim(Primitive):
    def __init__(self, random_state=0):
        super(InteractionFeaturesPrim, self).__init__(name="InteractionFeatures")
        self.id = 2
        self.gid = 18
        self.hyperparams = []
        self.type = "FeatureEngine"
        self.description = """Generate interaction features.

Only interaction features are produced: features that are products of at most degree distinct input features, i.e. terms with power of 2 or higher of the same input feature are excluded:"""
        self.scaler = PolynomialFeatures(interaction_only=True, include_bias=False)
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.scaler.fit(train_x)

        train_data_x = self.scaler.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x)
        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        self.scaler.fit(test_x)

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return test_data_x


class PCA_AUTO_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_AUTO_Prim, self).__init__(name="PCA_AUTO")
        self.id = 3
        self.gid = 19
        self.PCA_AUTO_Prim = []
        self.type = "FeatureEngine"
        self.description = """Principal component analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.

The solver is selected by a default ‘auto’ policy is based on X.shape and n_components: if the input data has fewer than 1000 features and more than 10 times as many samples, then the “covariance_eigh” solver is used. Otherwise, if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient “randomized” method is selected. Otherwise the exact “full” SVD is computed and optionally truncated afterwards.

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"""
        self.pca = PCA(svd_solver="auto")  # n_components=0.9
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cols = list(test_x.columns)
        self.pca.fit(test_x)

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return test_data_x


class IncrementalPCA_Prim(Primitive):
    def __init__(self, random_state=0):
        super(IncrementalPCA_Prim, self).__init__(name="IncrementalPCA")
        self.id = 5
        self.gid = 20
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = """Incremental principal components analysis (IPCA).

Linear dimensionality reduction using Singular Value Decomposition of the data, keeping only the most significant singular vectors to project the data to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.

Depending on the size of the input data, this algorithm can be much more memory efficient than a PCA, and allows sparse input.

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html"""
        self.hyperparams_run = {"default": True}
        self.pca = IncrementalPCA()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cols = list(test_x.columns)
        self.pca.fit(test_x)

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return test_data_x


class KernelPCA_Prim(Primitive):
    def __init__(self, random_state=0):
        super(KernelPCA_Prim, self).__init__(name="KernelPCA")
        self.id = 6
        self.gid = 21
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = """Kernel Principal component analysis (KPCA).

Non-linear dimensionality reduction through the use of kernels [1], see also Pairwise metrics, Affinities and Kernels.

[1] Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller. “Kernel principal component analysis.” International conference on artificial neural networks. Springer, Berlin, Heidelberg, 1997.

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html"""
        self.pca = KernelPCA(n_components=2)  # n_components=5
        self.accept_type = "c_t_krnl"
        self.random_state = random_state
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cols = list(test_x.columns)
        self.pca.fit(test_x)

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return test_data_x


class TruncatedSVD_Prim(Primitive):
    def __init__(self, random_state=0):
        super(TruncatedSVD_Prim, self).__init__(name="TruncatedSVD")
        self.id = 7
        self.gid = 22
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = """Dimensionality reduction using truncated SVD (aka LSA).

This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with sparse matrices efficiently.

In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html"""
        self.hyperparams_run = {"default": True}
        self.pca = TruncatedSVD(n_components=2)
        self.accept_type = "c_t_krnl"
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cols = list(test_x.columns)
        self.pca.fit(test_x)

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return test_data_x


class RandomTreesEmbeddingPrim(Primitive):
    def __init__(self, random_state=0):
        super(RandomTreesEmbeddingPrim, self).__init__(name="RandomTreesEmbedding")
        self.id = 8
        self.gid = 23
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = """An ensemble of totally random trees.

An unsupervised transformation of a dataset to a high-dimensional sparse representation. A datapoint is coded according to which leaf of each tree it is sorted into. Using a one-hot encoding of the leaves, this leads to a binary coding with as many ones as there are trees in the forest.

The dimensionality of the resulting representation is n_out <= n_estimators * max_leaf_nodes. If max_leaf_nodes == None, the number of leaf nodes is at most n_estimators * 2 ** max_depth.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html
"""
        self.hyperparams_run = {"default": True}
        self.pca = RandomTreesEmbedding(random_state=random_state)
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.pca.fit(train_x)
        cols = list(train_x.columns)

        train_data_x = self.pca.transform(train_x).toarray()
        new_cols = list(map(str, list(range(train_data_x.shape[1]))))
        train_data_x = pd.DataFrame(train_data_x, columns=new_cols)

        test_data_x = self.pca.transform(test_x).toarray()
        new_cols = list(map(str, list(range(test_data_x.shape[1]))))
        test_data_x = pd.DataFrame(test_data_x, columns=new_cols)
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        # self.pca.fit(test_x)

        test_data_x = self.pca.transform(test_x).toarray()
        new_cols = list(map(str, list(range(test_data_x.shape[1]))))
        test_data_x = pd.DataFrame(test_data_x, columns=new_cols)
        return test_data_x


class PCA_ARPACK_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_ARPACK_Prim, self).__init__(name="PCA_ARPACK")
        self.id = 4
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "ARPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver="arpack", n_components=2)
        self.accept_type = "c_t_arpck"
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x


class PCA_LAPACK_Prim(Primitive):
    def __init__(self, random_state=0):
        super(PCA_LAPACK_Prim, self).__init__(name="PCA_LAPACK")
        self.id = 3
        self.gid = 25
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "LAPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.pca = PCA(svd_solver="full")  # n_components=0.9
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x
