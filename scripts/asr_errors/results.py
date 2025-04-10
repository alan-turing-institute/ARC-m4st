import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_scores_path


def scale_scores(scores: np.ndarray, metric) -> np.ndarray:
    """
    Crude scaling of metrics to [0, 1] range.
    - BLASER: 1 - 5 XSTS scores to [0, 1]
    - COMET: 0 - 1 already
    - MetricX: 0 - 25 scores to [0, 1] and sign flip to higher is better
    """
    if "blaser" in metric:
        return (scores - 1) / 4
    if metric == "comet":
        return scores
    if metric == "metricx":
        return (25 - scores) / 25
    err = f"Unknown metric {metric}"
    raise ValueError(err)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    colors = {
        "metricx": "orange",
        "comet": "g",
        "blaser_text": "m",
        "blaser_audio": "b",
    }
    labels = {
        "metricx": "MetricX",
        "comet": "COMET",
        "blaser_text": "BLASER Text",
        "blaser_audio": "BLASER Audio",
    }

    blaser_comet_ratios = {}
    blaser_comet_xaxes = {}
    groups = ["perturbed", "merged", "random"]
    for i, group in enumerate(groups):
        metrics = ["comet", "metricx", "blaser_text", "blaser_audio"]
        scores = {}
        for metric in metrics:
            scores_path = get_scores_path(group, metric)
            scores[metric] = scale_scores(np.loadtxt(scores_path), metric)

        x = np.arange(len(scores["comet"])) / len(scores["comet"])
        x = list(reversed(x))
        xlabel = "Fraction of text replaced"
        for metric in metrics:
            ax[i].plot(
                x,
                scores[metric],
                label=labels[metric],
                color=colors[metric],
            )

        ax[i].set_xlabel(xlabel)
        ax[i].set_title(group.capitalize())

        df = pd.DataFrame(scores)
        df[xlabel] = x
        df.set_index(xlabel, inplace=True)
        df.to_csv(f"data/results/{group}_scaled_scores.csv")

        blaser_comet_ratios[group] = df["blaser_audio"] / df["comet"]
        blaser_comet_xaxes[group] = x

    ax[0].set_ylabel("Scaled Metric Score")
    ax[0].set_ylim(-0.1, 1.02)
    ax[0].set_xlim(-0.1, 1.02)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("data/results/results.png", dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for group in groups:
        ax.plot(blaser_comet_xaxes[group], blaser_comet_ratios[group], label=group)
    ax.set_xlabel("Fraction of text replaced")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.set_title("BLASER-Audio / COMET Score Ratios")
    fig.tight_layout()
    fig.savefig("data/results/blaser_comet_ratios.png", dpi=300)
