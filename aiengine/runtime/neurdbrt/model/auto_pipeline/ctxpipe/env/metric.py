import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, f1_score, mean_squared_error


class Metric:
    def evaluate(self, pred_y, test_y):
        pass


class AccuracyMetric(Metric):
    def __init__(self):
        super(AccuracyMetric, self).__init__()
        self.id = 1
        self.type = "Classifier"

    def evaluate(self, pred_y, test_y):
        if pred_y is None or test_y is None:
            return
        if isinstance(pred_y, pd.Series):
            pred_y = pred_y.values
        if isinstance(test_y, pd.Series):
            test_y = test_y.values

        self.score = accuracy_score(test_y, pred_y)
        return self.score


class F1Metric(Metric):
    def __init__(self):
        super(F1Metric, self).__init__()
        self.id = 2
        self.type = "Classifier"

    def evaluate(self, pred_y, test_y):
        if pred_y is None or test_y is None:
            return
        if isinstance(pred_y, pd.Series):
            pred_y = pred_y.values
        if isinstance(test_y, pd.Series):
            test_y = test_y.values

        self.score = f1_score(test_y, pred_y)
        return self.score


class AucMetric(Metric):
    def __init__(self):
        super(AucMetric, self).__init__()
        self.id = 3
        self.type = "Classifier"

    def evaluate(self, pred_y, test_y):
        if pred_y is None or test_y is None:
            return
        if isinstance(pred_y, pd.Series):
            pred_y = pred_y.values
        if isinstance(test_y, pd.Series):
            test_y = test_y.values

        fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
        self.score = metrics.auc(fpr, tpr)
        return self.score


class MseMetric(Metric):
    def __init__(self):
        super(MseMetric, self).__init__()
        self.id = 4
        self.type = "Regressor"

    def evaluate(self, pred_y, test_y):
        if pred_y is None or test_y is None:
            return
        if isinstance(pred_y, pd.Series):
            pred_y = pred_y.values
        if isinstance(test_y, pd.Series):
            test_y = test_y.values

        self.score = mean_squared_error(test_y, pred_y)
        return self.score
