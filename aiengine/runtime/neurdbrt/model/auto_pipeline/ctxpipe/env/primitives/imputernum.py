from copy import deepcopy

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .primitive import Primitive


def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x


class ImputerMean(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMean, self).__init__(name="ImputerMean")
        self.id = 1
        self.gid = 1
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by mean."
        )
        self.imp = SimpleImputer()
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isnull().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)
        self.imp.fit(num_trainX)
        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],
            axis=1,
        )

        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_testX, num_testX = catch_num(test_x)
        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return test_data_x


class ImputerMedian(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMedian, self).__init__(name="ImputerMedian")
        self.id = 2
        self.gid = 2
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by median."
        )
        self.imp = SimpleImputer(strategy="median")
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isnull().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)
        self.imp.fit(num_trainX)
        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],
            axis=1,
        )
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_testX, num_testX = catch_num(test_x)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return test_data_x


class ImputerNumPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerNumPrim, self).__init__(name="ImputerNumMode")
        self.id = 4
        self.gid = 4
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by mode."
        )
        self.imp = SimpleImputer(strategy="most_frequent")
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isnull().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)
        self.imp.fit(num_trainX)

        cols = list(num_trainX.columns)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],
            axis=1,
        )

        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )

        return train_data_x, test_data_x

    def transform_x(self, test_x):
        cat_testX, num_testX = catch_num(test_x)
        cols = list(num_testX.columns)
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return test_data_x
