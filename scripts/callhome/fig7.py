import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 20


def colour_plot(data, a, b):
    # Define bins
    bins = np.linspace(min(data), max(data), 60)  # 30 bins

    # Compute histogram (get counts and bin edges)
    counts, bin_edges = np.histogram(data, bins=bins)

    # Assign colors to bins based on boundaries
    colors = []
    for i in range(len(bin_edges) - 1):
        if bin_edges[i + 1] < a:  # Below boundary 1
            colors.append("red")
        elif bin_edges[i] >= b:  # Above boundary 2
            colors.append("green")
        else:  # Between boundaries
            colors.append("blue")

    # Plot histogram manually using `bar()`
    plt.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    # Create a custom legend
    legend_patches = [
        mpatches.Patch(color="red", label=f"< {a:.1f}"),
        mpatches.Patch(color="blue", label=f"{a:.1f} to {b:.1f}"),
        mpatches.Patch(color="green", label=f"> {b:.1f}"),
    ]
    plt.legend(handles=legend_patches)

    # plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


df = pd.read_json("metric_evaluation.json")
df["original_length"] = df["originals"].transform(len)

keep_longer_than = 30
eps = 1e-6
metric_names = {
    "COMETQEScore",
    "BLASERRefScore",
    "BLASERQEScore",
    "ChrFScore",
    "SacreBLEUScore",
}

for metric in metric_names:
    df_longer_than = df[df["original_length"] >= keep_longer_than]

    # Compute quartile boundaries
    a = np.percentile(df_longer_than[metric], 20)
    b = np.percentile(df_longer_than[metric], 85)

    plt.cla()
    colour_plot(df_longer_than[metric], a, b)
    plt.xlabel(metric)
    plt.gcf().tight_layout()
    plt.savefig(f"hist_{metric}.png")

    df_0 = df_longer_than[df_longer_than[metric] < a]
    df_1 = df_longer_than[(df_longer_than[metric] >= a) & (df_longer_than[metric] < b)]
    df_2 = df_longer_than[df_longer_than[metric] > b]

    for idx, selected_df in zip(range(3), [df_0, df_1, df_2], strict=False):
        _sample = selected_df.sample(5)[["originals", "references", "translations"]]
        band_orig, band_ref, band_tra = (
            _sample["originals"],
            _sample["references"],
            _sample["translations"],
        )
        _sample.to_csv(f"examples_{metric}_{idx}.csv", index=False)
