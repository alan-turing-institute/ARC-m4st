"""
Score a source and translation group with MetricX QE.
"""

from utils import get_group, get_group_hypotheses, get_group_sources, get_scores_path

from m4st.metrics import TranslationDataset
from m4st.metrics.metricx import MetricXScore

if __name__ == "__main__":
    group = get_group()
    sources = get_group_sources(group)
    hypotheses = get_group_hypotheses(group)

    metricx = MetricXScore("google/metricx-24-hybrid-large-v2p6", qe=True)

    scores = metricx.get_scores(
        TranslationDataset(prediction=hypotheses, source=sources)
    )

    scores_path = get_scores_path(group, "metricx")
    with open(scores_path, "w") as f:
        for score in scores:
            f.write(f"{score}\n")
