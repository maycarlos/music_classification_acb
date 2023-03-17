import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def pairplot(df,target_variable : str,  save_image : bool):
    yeah = np.random.randint(0, 100, 5)

    ax = sns.pairplot(df, diag_kind="kde", vars = df.columns[yeah], hue = target_variable)
    ax._legend.set_title("Classes")
    sns.move_legend(ax, bbox_to_anchor=(1.04, 0.5), loc="center left", fancybox=True, shadow=True )
    plt.suptitle("Classes distributions and Correlation")
    plt.tight_layout()
    if save_image:
        plt.savefig("../reports/figures/classes_pairplot.png", format = "png")
    plt.show()

def class_distribution(df,target_variable : str, save_image : bool):
    n = df[target_variable].nunique()
    colors = [f"C{x}" for x in range(n)]
    df[target_variable].value_counts().plot(kind = "bar", rot = 30, color = colors)

    plt.title("Number of samples for each class")
    plt.xlabel("Classes")
    plt.ylabel("Total")
    plt.tight_layout()
    if save_image:
        plt.savefig("../reports/figures/classes_distribution.png", format = "png")
    plt.show()