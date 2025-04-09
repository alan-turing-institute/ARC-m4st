import re

with open("data/source_corrected.txt") as f:
    source = f.read()

with open("data/alt_source.txt") as f:
    alt_source = f.read()

words = re.split(r"(\s|,|\.)", source)
alt_words = alt_source.split()


for i in range(len(words)):
    n_alt = min(len(words) - i - 1, len(alt_words))
    perturbed_source = "".join(words[:i]).strip() + " " + " ".join(alt_words[:n_alt])
    with open(f"data/source_merged/merged_source_{i}.txt", "w") as f:
        f.write(perturbed_source)
