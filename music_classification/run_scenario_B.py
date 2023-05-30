from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .data.process_data import clean_dataset, define_pipeline
from .utils.dotenv_loading import ENV_CONFIG
from .visualization.visualize import (class_distribution, interpret_steps,
                                      pairplot)

SAVE = True

SEED = int(ENV_CONFIG["SEED"])
np.random.default_rng(seed=SEED)


df_raw = pd.read_csv(Path(ENV_CONFIG["DATA_FOLDER"]), dtype_backend="pyarrow")

# Data cleaning
df = clean_dataset(df_raw)

# EDA
print(f"Number of missing values in the dataset is {df.isnull().sum().sum()}.")

pairplot(data=df, target_variable="genre", save_image=SAVE)

class_distribution(df, target_variable="genre", save_image=SAVE)

# Data Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
input_features = df.columns[:-1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=SEED, test_size=0.2,
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


lof = LocalOutlierFactor()
yhet = lof.fit_predict(X_train)
mask = yhet != -1
X_train, y_train = X_train[mask, :], y_train[mask]

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

pipeline = define_pipeline()

X_train = pipeline.fit_transform(X_train, y_train)
X_test = pipeline.transform(X_test)


interpret_steps(pipeline=pipeline, features=input_features)

X_train, X_grid, y_train, y_grid = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED
)


# Modelling
## Gridsearch for hyperparameters
lg_grid = GridSearchCV(
    OneVsRestClassifier(LogisticRegression()),
    param_grid={"estimator__C": [100, 10, 1.0, 0.1, 0.01]},
    scoring="f1_weighted",
    n_jobs=4,
    cv=5,
)

knn_grid = GridSearchCV(
    OneVsRestClassifier(KNeighborsClassifier()),
    param_grid={"estimator__n_neighbors": np.arange(5, 105, 5)},
    scoring="f1_weighted",
    n_jobs=4,
    cv=5,
)


rf_grid = GridSearchCV(
    OneVsRestClassifier(RandomForestClassifier()),
    param_grid={"estimator__n_estimators": np.arange(10, 101, 5)},
    scoring="f1_weighted",
    n_jobs=4,
    cv=5,
)


svm_grid = GridSearchCV(
    OneVsRestClassifier(SVC()),
    param_grid={"estimator__C": np.arange(1, 51, 5)},
    scoring="f1_weighted",
    n_jobs=4,
    cv=5,
)

lg_grid.fit(X_grid, y_grid)
knn_grid.fit(X_grid, y_grid)
rf_grid.fit(X_grid, y_grid)
svm_grid.fit(X_grid, y_grid)


models = [
    "LogReg",
    "KNN",
    "LDA",
    "RF",
    "NB",
    "SVM",
]

scores = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro"),
    "f1_weighted": make_scorer(f1_score, average="macro"),
    "mcc": make_scorer(matthews_corrcoef),
}


lg_crossval_scores = cross_validate(
    OneVsRestClassifier(LogisticRegression(C=0.1)),
    X_train,
    y_train,
    cv=5,
    scoring=scores,
    n_jobs=8,
)

knn_crossval_scores = cross_validate(
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
    X_train,
    y_train,
    cv=5,
    scoring=scores,
    n_jobs=8,
)

lda_crossval_scores = cross_validate(
    OneVsRestClassifier(LDA()), X_train, y_train, cv=5, scoring=scores, n_jobs=8,
)

rf_crossval_scores = cross_validate(
    OneVsRestClassifier(RandomForestClassifier(n_estimators=50)),
    X_train,
    y_train,
    cv=5,
    scoring=scores,
    n_jobs=8,
)

svm_crossval_scores = cross_validate(
    OneVsRestClassifier(SVC(C=1)), X_train, y_train, cv=5, scoring=scores, n_jobs=8,
)

nb_crossval_scores = cross_validate(
    OneVsRestClassifier(GaussianNB()), X_train, y_train, cv=5, scoring=scores, n_jobs=8,
)


fig, axs = plt.subplots(3, 2, figsize=(12, 9))
axs = axs.flatten()

for ax, score in zip(axs, scores.keys()):
    test_score = "test_" + score

    cross_dict = {
        "LogReg": lg_crossval_scores[test_score],
        "kNN": knn_crossval_scores[test_score],
        "LDA": lda_crossval_scores[test_score],
        "RForest": rf_crossval_scores[test_score],
        "SVM": svm_crossval_scores[test_score],
        "NBayes": nb_crossval_scores[test_score],
    }

    cross_val_scores = pd.DataFrame(cross_dict)

    sns.boxplot(cross_val_scores, ax=ax)

    ax.set(
        title=f"CrossVal {score.capitalize()} ",
        xlabel="Classifiers",
        ylabel=f"{score.capitalize()} Value",
    )

plt.tight_layout()
plt.show()


model_instances = [
    OneVsRestClassifier(LogisticRegression(C=1.0)),
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
    OneVsRestClassifier(LDA()),
    OneVsRestClassifier(RandomForestClassifier(n_estimators=90)),
    OneVsRestClassifier(SVC(C=46)),
    OneVsRestClassifier(GaussianNB()),
]


fitted_models = [*map(lambda x: x.fit(X_train, y_train), model_instances)]
predictions = [*map(lambda x: x.predict(X_test), fitted_models)]

single_test = pd.DataFrame()
single_test.index = scores.keys()

for model, pred in zip(models, predictions):
    pred_score = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted"),
        "recall": recall_score(y_test, pred, average="weighted"),
        "f1": f1_score(y_test, pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, pred),
    }

    single_test[model] = pred_score.values()

single_test.index = single_test.index.str.capitalize()

single_test.to_excel("multiclass_single_test.xlsx")
print(single_test.round(decimals=3))
