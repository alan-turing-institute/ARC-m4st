"""
Create a sequence of sources with a varying number of words removed at random.
"""

import random

import numpy as np
from utils import get_group_src_path, get_original_source

if __name__ == "__main__":
    source = get_original_source()
    output_path = get_group_src_path("random")

    words = np.array(source.split())
    order = list(range(len(words)))
    random.shuffle(order)  # in-place shuffle

    for i in range(len(words)):
        keep = sorted(order[: (i + 1)])
        perturbed_source = " ".join(words[keep])
        with open(f"{output_path}_{i}.txt", "w") as f:
            f.write(perturbed_source)
