"""
Create a sequence of sources with varying number of words replaced by text from an
alternative source.
"""

import re

from utils import get_alt_source, get_group_src_path, get_original_source

if __name__ == "__main__":
    source = get_original_source()
    alt_source = get_alt_source()
    output_path = get_group_src_path("merged")

    words = re.split(r"(\s|,|\.)", source)  # separate into words and punctuation
    alt_words = alt_source.split()  # separate into words

    for i in range(len(words)):
        n_alt = min(len(words) - i - 1, len(alt_words))
        perturbed_source = (
            "".join(words[:i]).strip() + " " + " ".join(alt_words[:n_alt])
        )
        with open(f"{output_path}_{i}.txt", "w") as f:
            f.write(perturbed_source)
