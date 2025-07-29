import pytest

from ctxpipe.out import build_context, render_script


def test_build_context():
    context = build_context(["ImputerMean", "StandardScaler", "RandomTreesEmbedding"])
    expected = {
        "meta": {
            "head": """import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
""",
            "global": """def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x


class Primitive:
    def __init__(self, name="blank"):
        self.name = name
        self.description = "No-op"
        self.type = "blank"

    def transform(
        self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_x, test_x

    def __repr__(self) -> str:
        return f"<{self.name}>"
""",
        },
        "components": [
            {
                "head": """from sklearn.impute import SimpleImputer""",
                "global": """class ImputerMean(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMean, self).__init__(name="ImputerMean")
        self.type = "ImputerNum"
        self.imp = SimpleImputer()

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
""",
                "local": """train_x, test_x = ImputerMean().transform(train_x, test_x, train_y)
""",
            },
            {
                "head": """from sklearn.preprocessing import StandardScaler""",
                "global": """class StandardScalerPrim(Primitive):
    def __init__(self, random_state=0):
        super(StandardScalerPrim, self).__init__(name="StandardScaler")
        self.type = "FeaturePreprocessing"
        self.scaler = StandardScaler()

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
""",
                "local": """train_x, test_x = StandardScalerPrim().transform(train_x, test_x, train_y)
""",
            },
            {
                "head": """from sklearn.ensemble import RandomTreesEmbedding""",
                "global": """class RandomTreesEmbeddingPrim(Primitive):
    def __init__(self, random_state=0):
        super(RandomTreesEmbeddingPrim, self).__init__(name="RandomTreesEmbedding")
        self.type = "FeatureEngine"
        self.pca = RandomTreesEmbedding(random_state=random_state)

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
""",
                "local": """train_x, test_x = RandomTreesEmbeddingPrim().transform(train_x, test_x, train_y)
""",
            },
        ],
        "model": {
            "head": """from sklearn.linear_model import LogisticRegression""",
            "global": """class LogisticRegressionPrim(Primitive):
    def __init__(self):
        super(LogisticRegressionPrim, self).__init__(name="LogisticRegression")
        self.type = "Classifier"

    def predict(self, train_x, test_x, train_y):
        model = LogisticRegression(solver="liblinear", random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y
""",
            "local": """pred_y = LogisticRegressionPrim().predict(train_x, test_x, train_y)
""",
        },
    }
    assert context == expected


def test_render_script():
    context = build_context(["ImputerMean", "StandardScaler", "RandomTreesEmbedding"])
    script = render_script(context)

    with open("test_pipeline_out.py", "w") as f:
        f.write(script)

    assert script


def test_render_script_with_extra():
    context = build_context(["ImputerMean", "StandardScaler", "RandomTreesEmbedding"])
    script = render_script(context, label_column="label", data_dir="data")

    assert script
