from typing import List

from ctxpipe.env.metric import *
from ctxpipe.env.primitives import *
import env

selected_prim: Primitive = LogisticRegressionPrim()

imputernums: List[Primitive] = [ImputerMean(), ImputerMedian(), ImputerNumPrim()]
encoders: List[Primitive] = [
    NumericDataPrim(),
    LabelEncoderPrim(),
    OneHotEncoderPrim(),
]
fpreprocessings: List[Primitive] = [
    MinMaxScalerPrim(),
    MaxAbsScalerPrim(),
    RobustScalerPrim(),
    StandardScalerPrim(),
    QuantileTransformerPrim(),
    PowerTransformerPrim(),
    NormalizerPrim(),
    KBinsDiscretizerOrdinalPrim(),
    Primitive(),
]
fengines: List[Primitive] = [
    PolynomialFeaturesPrim(),
    InteractionFeaturesPrim(),
    PCA_AUTO_Prim(),
    IncrementalPCA_Prim(),
    KernelPCA_Prim(),
    TruncatedSVD_Prim(),
    RandomTreesEmbeddingPrim(),
    Primitive(),
]
fselections: List[Primitive] = [VarianceThresholdPrim(), Primitive()]

logic_pipeline_1 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeaturePreprocessing",
    "FeatureEngine",
    "FeatureSelection",
]
logic_pipeline_2 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeaturePreprocessing",
    "FeatureSelection",
    "FeatureEngine",
]
logic_pipeline_3 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeatureEngine",
    "FeatureSelection",
    "FeaturePreprocessing",
]
logic_pipeline_4 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeatureEngine",
    "FeaturePreprocessing",
    "FeatureSelection",
]
logic_pipeline_5 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeatureSelection",
    "FeatureEngine",
    "FeaturePreprocessing",
]
logic_pipeline_6 = [
    "ImputerNum",
    "ImputerCat",
    "Encoder",
    "FeatureSelection",
    "FeaturePreprocessing",
    "FeatureEngine",
]

lpipelines = [
    logic_pipeline_1,
    logic_pipeline_2,
    logic_pipeline_3,
    logic_pipeline_4,
    logic_pipeline_5,
    logic_pipeline_6,
]

predictors = [
    RandomForestClassifierPrim(),
    AdaBoostClassifierPrim(),
    BaggingClassifierPrim(),
    BernoulliNBClassifierPrim(),
    DecisionTreeClassifierPrim(),
    ExtraTreesClassifierPrim(),
    GaussianNBClassifierPrim(),
    GradientBoostingClassifierPrim(),
    KNeighborsClassifierPrim(),
    LinearDiscriminantAnalysisPrim(),
    selected_prim,
    MLPClassifierPrim(),
    NearestCentroidPrim(),
    PassiveAggressiveClassifierPrim(),
    RidgeClassifierPrim(),
    RidgeClassifierCVPrim(),
    SGDClassifierPrim(),
    SVCPrim(),
    # GaussianProcessClassifierPrim(),
    # ComplementNBClassifierPrim(),
    # LogisticRegressionCVPrim(),
    # MultinomialNBPrim(),
    # QuadraticDiscriminantAnalysisPrim(),
]

num_imputernums: int = len(imputernums)
num_encoders: int = len(encoders)
num_fpreprocessings: int = len(fpreprocessings)
num_fengines: int = len(fengines)
num_fselections: int = len(fselections)

single_action_dim: int = max(
    [
        num_imputernums,
        num_encoders,
        num_fpreprocessings,
        num_fengines,
        num_fselections,
    ]
)

num_predictors = len(predictors)
num_lpipelines: int = len(lpipelines)

metrics = [
    AccuracyMetric(),
    F1Metric(),
    AucMetric(),
    MseMetric(),
]

dtype_id_map = {
    "interval[float64]": 4,
    "uint8": 1,
    "uint16": 1,
    "int64": 1,
    "int": 1,
    "int32": 1,
    "int16": 1,
    "np.int32": 1,
    "np.int64": 1,
    "np.int": 1,
    "np.int16": 1,
    "float64": 2,
    "float": 2,
    "float32": 2,
    "float16": 2,
    "np.float32": 2,
    "np.float64": 2,
    "np.float": 2,
    "np.float16": 2,
    "str": 3,
    "Category": 4,
    "object": 4,
    "bool": 5,
}
