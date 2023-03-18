from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ..utils.dotenv_loading import ENV_CONFIG
from sklearn.decomposition import PCA

from ..utils.types_ import DataFrame

# Set the random seed 
SEED = int(ENV_CONFIG["SEED"])
np.random.default_rng(SEED)

# Set the plotting style for vizualization
plt.style.use(ENV_CONFIG["PLOT_STYLE"])
sns.set_palette(ENV_CONFIG["PALETTE"])

# Set the default save path for the images obtained
default_save = Path(ENV_CONFIG["FIGURES_FOLDER"])


def pairplot(
    data: DataFrame,
    target_variable: str,
    save_image: bool = True,
    save_location: Path = default_save,
):

    yeah = np.random.randint(0, 100, 5)

    ax = sns.pairplot(
        data, diag_kind="kde", vars=data.columns[yeah], hue=target_variable
    )
    ax._legend.set_title("Classes")
    sns.move_legend(
        ax, bbox_to_anchor=(1.04, 0.5), loc="center left", fancybox=True, shadow=True
    )
    plt.suptitle("Classes distributions and Correlation")
    plt.tight_layout()
    if save_image:
        plt.savefig(save_location / "classes_pairplot.png", format="png")
    plt.show()


def class_distribution(
    data: DataFrame,
    target_variable: str,
    save_image: bool = True,
    save_location: Path = default_save,
):
    n = data[target_variable].nunique()
    colors = [f"C{x}" for x in range(n)]
    data[target_variable].value_counts(normalize = True).plot(kind="bar", rot=30, color=colors)

    plt.title("Number of samples for each class")
    plt.xlabel("Classes")
    plt.ylabel("Total")
    plt.tight_layout()
    if save_image:
        plt.savefig(save_location / "classes_distribution.png", format="png")
    plt.show()


def feature_importance(
    scores,
    features, 
    save_image : bool = True,
    save_location : Path = default_save,
):

    score_sorting = np.argsort(scores)[::-1]
    n = len(scores)

    y_pos = np.arange(1, n + 1)

    kbest_features = (
        features
        .str.replace("stft_", "")
        .str.replace("_max", "")
        .str.replace("mean_0_", "")
    )

    ax = plt.subplot()

    colors = [f"C{x}" for x in range(10)]

    ax.bar(y_pos, scores[score_sorting], color = colors, label = kbest_features[score_sorting])

    ax.set(
        title=f"{n} Best features",
        xticks=y_pos,
        ylabel="Feature importance values",
        xlabel="Features"
    )

    plt.legend(
        title = "Features",
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        fancybox=True,
        shadow=True
    )

    if save_image:
        plt.savefig(save_location / "best_k_features.png", format = "png")
    plt.show()


def scree_plot(
    pc_values,
    pca_variance,
    save_image : bool = True,
    save_location : Path = default_save
):
    
    plt.plot(pc_values, pca_variance, 'o-', linewidth=2, color='blue')
    plt.xticks([*range(1,11)])
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

    if save_image:
        plt.savefig(save_location / "scree_plot.png", format = "png")

    plt.show()

def interpret_steps(pipeline, features):

    pipeline_len = len(pipeline)

    vt_mask = pipeline["variance_threshold"].get_support()
    vt_features = features[vt_mask]

    kbest_mask = pipeline["k_best"].get_support()
    kbest_features = vt_features[kbest_mask]
    scores = pipeline["k_best"].scores_[kbest_mask]

    feature_importance(scores, kbest_features)

    if isinstance(pipeline[pipeline_len -1], PCA):
        
        pc_values = np.arange(pipeline["pca"].n_components_) + 1
        pca_variance = pipeline["pca"].explained_variance_ratio_ * 100

        scree_plot(pc_values, pca_variance)


