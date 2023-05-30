from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import set_config
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .data.process_data import clean_dataset, define_pipeline
from .utils.dotenv_loading import ENV_CONFIG
from .visualization.visualize import (class_distribution, interpret_steps,
                                      pairplot, plot_confusion, plot_pca,
                                      plot_rocs)

SAVE = True

SEED = int(ENV_CONFIG["SEED"])
np.random.default_rng(seed=SEED)


set_config(display="diagram")

# Load the data
df_raw = pd.read_csv(Path(ENV_CONFIG["DATA_FOLDER"]), dtype_backend="pyarrow")
df = clean_dataset(df_raw, "A", "country")

# Quick EDA
print(f"Number of missing values in the dataset is {df.isnull().sum().sum()}")
pairplot(data=df, target_variable="genre", save_image=SAVE)
class_distribution(df, target_variable="genre", save_image=SAVE)

# Data Splitting
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

input_features = df.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=SEED, test_size=0.2,
)

# Encode Target variable
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Remove Outliers using LOF
lof = LocalOutlierFactor()
yhet = lof.fit_predict(X_train)
mask = yhet != -1
X_train, y_train = X_train[mask, :], y_train[mask]


# Balance classes distribution for training
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Setup pipeline
pipeline = define_pipeline()
X_train = pipeline.fit_transform(X_train, y_train)
X_test = pipeline.transform(X_test)

# Interpret the pipeline steps
interpret_steps(pipeline=pipeline, features=input_features)
plot_pca(X_train, y_train, le)

X_train, X_grid, y_train, y_grid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED,
)

# Grid search 
lg_grid = GridSearchCV(
    LogisticRegression(),
    param_grid={"C": [100, 10, 1.0, 0.1, 0.01]},
    scoring="f1",
    n_jobs=4,
    cv=5,
)

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": np.arange(5, 105, 5)},
    scoring="f1",
    n_jobs=4,
    cv=5,
)

rf_grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid={"n_estimators": np.arange(10, 101, 5)},
    scoring="f1",
    n_jobs=4,
    cv=5,
)

svm_grid = GridSearchCV(
    SVC(), param_grid={"C": np.arange(1, 50, 5)}, scoring="f1", n_jobs=4, cv=5
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
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "mcc": make_scorer(matthews_corrcoef),
}

lg_crossval_scores = cross_validate(
    LogisticRegression(**lg_grid.best_params_),
    X_train,
    y_train,
    cv=50,
    scoring=scores,
    n_jobs=8,
)

knn_crossval_scores = cross_validate(
    KNeighborsClassifier(**knn_grid.best_params_),
    X_train,
    y_train,
    cv=50,
    scoring=scores,
    n_jobs=8,
)

lda_crossval_scores = cross_validate(
    LDA(), X_train, y_train, cv=50, scoring=scores, n_jobs=8,
)

rf_crossval_scores = cross_validate(
    RandomForestClassifier(**rf_grid.best_params_),
    X_train,
    y_train,
    cv=50,
    scoring=scores,
    n_jobs=8,
)

svm_crossval_scores = cross_validate(
    SVC(**svm_grid.best_params_), X_train, y_train, cv=50, scoring=scores, n_jobs=8,
)

nb_crossval_scores = cross_validate(
    GaussianNB(), X_train, y_train, cv=50, scoring=scores, n_jobs=8,
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
    LogisticRegression(**lg_grid.best_params_),
    KNeighborsClassifier(**knn_grid.best_params_),
    LDA(),
    RandomForestClassifier(**rf_grid.best_params_),
    SVC(**svm_grid.best_params_),
    GaussianNB(),
]

fitted_models = map(lambda x: x.fit(X_train, y_train), model_instances)
predictions = [*map(lambda x: x.predict(X_test), fitted_models)]


plot_rocs(models, predictions, y_test)

plot_confusion(models, predictions, y_test)

single_test = pd.DataFrame()
single_test.index = scores.keys()

for model, pred in zip(models, predictions):
    pred_score = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "mcc": matthews_corrcoef(y_test, pred),
    }

    single_test[model] = pred_score.values()

single_test.index = single_test.index.str.capitalize()
single_test.to_excel("binary_single_test.xlsx")
print(single_test.round(decimals=3))
