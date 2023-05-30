import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from ..utils.dotenv_loading import ENV_CONFIG
from ..utils.types_ import DataFrame

# Set the random seed
SEED = int(ENV_CONFIG["SEED"])

random.seed(SEED)
np.random.default_rng(SEED)

# Set the plotting style for vizualization
plt.rcParams["figure.figsize"] = (12, 6)
plt.style.use(ENV_CONFIG["PLOT_STYLE"])
sns.set_palette(ENV_CONFIG["PALETTE"])

# Set the default save path for the images obtained
default_save = Path(ENV_CONFIG["FIGURES_FOLDER"])


def pairplot(
    data: DataFrame,
    target_variable: Optional[str] = None,
    save_image: bool = True,
    save_location: Path = default_save,
    figure_name: str = "classes_pairplot.png",
    palette: Optional[list[str]] = None,
):
    if target_variable:
        n = data[target_variable].nunique()
        palette = [f"C{x}" for x in range(n)]
        # palette ={"blues": "C1", "others": "C0"}

    random_cols = random.sample(range(len(data.columns)), k=5)

    ax = sns.pairplot(
        data,
        diag_kind="kde",
        vars=data.columns[random_cols],
        hue=target_variable,
        palette=palette,
    )

    # plt.legend()
    # ax.add_legend()

    sns.move_legend(
        ax,
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        fancybox=True,
        shadow=True,
        title="Classes",
    )

    plt.suptitle("Classes distributions and Correlation")
    plt.tight_layout()
    if save_image:
        plt.savefig(save_location / figure_name, format="png")
    plt.show()


def class_distribution(
    data: DataFrame,
    target_variable: str,
    save_image: bool = True,
    save_location: Path = default_save,
    figure_name: str = "classes_distribution.png",
):
    # n = data[target_variable].nunique()
    # colors = [f"C{x}" for x in range(n)]
    data[target_variable].value_counts(normalize=True).plot(
        kind="bar",
        rot=30,
    )

    plt.title("Number of samples for each class")
    plt.xlabel("Classes")
    plt.ylabel("Total")

    plt.tight_layout()

    if save_image:
        plt.savefig(save_location / figure_name, format="png")
    plt.show()


def feature_importance(
    scores,
    features,
    save_image: bool = True,
    save_location: Path = default_save,
    figure_name: str = "best_k_features.png",
):

    score_sorting = np.argsort(scores)[::-1]
    n = len(scores)

    y_pos = np.arange(1, n + 1)

    kbest_features = (
        features.str.replace("stft_", "")
        .str.replace("_max", "")
        .str.replace("mean_0_", "")
    )

    ax = plt.subplot()

    colors = [f"C{x}" for x in range(10)]

    ax.bar(
        y_pos,
        scores[score_sorting],
        color=colors,
        label=kbest_features[score_sorting],
    )

    ax.set(
        title=f"{n} Best features",
        xticks=y_pos,
        ylabel="Feature importance values",
        xlabel="Features",
    )

    plt.legend(
        title="Features",
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    if save_image:
        plt.savefig(save_location / figure_name, format="png")

    plt.show()


def scree_plot(
    pc_values, pca_variance, save_image: bool = True, save_location: Path = default_save,
    figure_name: str = "pca_scree_plot.png",
):

    plt.plot(pc_values, pca_variance, "o-", linewidth=2, color="blue")
    plt.xticks([*range(1, len(pca_variance) + 1)])
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")

    plt.tight_layout()

    if save_image:
        plt.savefig(save_location / "scree_plot.png", format="png")

    plt.show()


def interpret_steps(pipeline, features):

    pipeline_len = len(pipeline)

    vt_mask = pipeline["variance_threshold"].get_support()
    vt_features = features[vt_mask]

    kbest_mask = pipeline["k_best"].get_support()
    kbest_features = vt_features[kbest_mask]
    scores = pipeline["k_best"].scores_[kbest_mask]

    feature_importance(scores, kbest_features)

    if isinstance(pipeline[pipeline_len - 1], PCA):

        pc_values = np.arange(pipeline["pca"].n_components_) + 1
        pca_variance = pipeline["pca"].explained_variance_ratio_ * 100

        scree_plot(pc_values, pca_variance)


def plot_rocs(models, predictions, y_true, save_image: bool = True,save_location: Path = default_save,figure_name : str = "roc_curve.png",):

    fig, axs = plt.subplots(2,3, figsize = (12,8))
    axs = axs.flatten()

    for model, prediction, ax in zip(models,predictions,axs):
        viz = RocCurveDisplay.from_predictions(
            y_pred = prediction,
            y_true = y_true,
            name =f"{model} ROC Curve",
            lw=3,
            ax=ax,)
        
        ax.set_title(model)
        
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    if save_image:
        plt.savefig(save_location / figure_name, format = "png")
    fig.suptitle("ROC Curves")
    plt.tight_layout()
    plt.show()

def plot_pca(X_pca, y_pca, encoder, save_image: bool  = True, save_location : Path = default_save,figure_name : str = "pca_plot.png"):
    ax = plt.subplot()

    unique_labels = np.unique(y_pca)

    for i, lab in enumerate(unique_labels):
        ax.scatter(
            X_pca.iloc[y_pca == lab, 0],
            X_pca.iloc[y_pca == lab, 1],
            c=f"C{i}",
            label=encoder.inverse_transform([lab]).item().capitalize(),
            s=30,
        )

    if save_image:
        plt.savefig(save_location / figure_name, format = "png")

    plt.legend(loc="best")
    plt.show()



def plot_confusion(models, predictions, y_true, save_image : bool = True, save_location: Path = default_save, figure_name : str = "confusion_matrices.png"):
    fig, axs = plt.subplots(2,3, figsize = (12,8))
    axs = axs.flatten()

    for model, predicition, ax in zip(models,predictions,axs):
        conf_matrix = confusion_matrix(y_true = y_true, y_pred = predicition, normalize = "all")
        sns.heatmap(conf_matrix, annot=True, ax=ax)
        ax.set_title(model)

    if save_image:
        plt.savefig(save_location / figure_name, format = "png")


    fig.suptitle("Confusion matrices for each classifier")

    plt.tight_layout()
    plt.show()

