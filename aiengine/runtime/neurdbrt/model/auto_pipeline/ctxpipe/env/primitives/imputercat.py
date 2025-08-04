import pandas as pd
from sklearn.impute import SimpleImputer

from .primitive import Primitive


def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x


class ImputerCatPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerCatPrim, self).__init__(name="ImputerCatMode")
        self.id = 1
        self.gid = 5
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
        self.imp.fit(cat_trainX)
        cols = list(cat_trainX.columns)
        cat_trainX = self.imp.fit_transform(cat_trainX.reset_index(drop=True))
        cat_trainX = pd.DataFrame(cat_trainX).reset_index(drop=True).infer_objects()
        cols = ["col_" + str(i) for i in cat_trainX.columns]
        cat_trainX.columns = cols
        cat_trainX = cat_trainX.reset_index(drop=True)
        num_trainX = num_trainX.reset_index(drop=True)

        train_data_x = pd.concat([cat_trainX, num_trainX], axis=1)
        cols = list(cat_testX.columns)
        cat_testX = self.imp.fit_transform(cat_testX.reset_index(drop=True))
        cat_testX = pd.DataFrame(cat_testX).reset_index(drop=True).infer_objects()
        cols = ["col_" + str(i) for i in cat_testX.columns]
        cat_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x
