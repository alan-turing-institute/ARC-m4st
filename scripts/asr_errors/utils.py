import argparse
import os

import torch
import torchaudio


def get_original_source() -> str:
    """Get original source text."""
    with open("data/inputs/source_corrected.txt") as f:
        return f.read()


def get_reference() -> str:
    """Get reference text."""
    with open("data/inputs/reference.txt") as f:
        return f.read()


def get_alt_source() -> str:
    """Get alternative source text (used for 'merged' group)."""
    with open("data/inputs/alt_source.txt") as f:
        return f.read()


def get_group_src_dir(group: str) -> str:
    """Get source directory for the specified group."""
    src_dir = f"data/source_{group}"
    if not os.path.isdir(src_dir):
        os.makedirs(src_dir)
    return src_dir


def get_group_mt_dir(group: str) -> str:
    """Get machine translation directory for the specified group."""
    mt_dir = f"data/translation_{group}"
    if not os.path.isdir(mt_dir):
        os.makedirs(mt_dir)
    return mt_dir


def get_group_src_path(group: str) -> str:
    """Get source path for the specified group."""
    src_dir = get_group_src_dir(group)
    return f"{src_dir}/{group}_source"


def get_group_mt_path(group: str) -> str:
    """Get machine translation path for the specified group."""
    mt_dir = get_group_mt_dir(group)
    return f"{mt_dir}/{group}_translation"


def get_group_sources(group: str) -> list[str]:
    """Get sources from the specified group."""
    sources = []
    src_dir = get_group_src_dir(group)
    src_path = get_group_src_path(group)
    n_files = len(os.listdir(src_dir))
    for i in range(n_files):
        with open(f"{src_path}_{i}.txt") as f:
            sources.append(f.read())
    return sources


def get_group_hypotheses(group: str) -> list[str]:
    """Get hypotheses from the specified group."""
    hypotheses = []
    mt_dir = get_group_mt_dir(group)
    mt_path = get_group_mt_path(group)
    n_files = len(os.listdir(mt_dir))
    for i in range(n_files):
        with open(f"{mt_path}_{i}.txt") as f:
            hypotheses.append(f.read())
    return hypotheses


def get_audio_path() -> str:
    """Get path to the audio file."""
    return "data/inputs/source.wav"


def get_source_audio() -> tuple[torch.tensor, int]:
    """Get source audio."""
    return torchaudio.load(get_audio_path())


def get_group():
    """Get group from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "group",
        type=str,
        choices=["random", "perturbed", "merged"],
        help="Source group to use ('random', 'perturbed', or 'merged')",
    )
    args = parser.parse_args()
    return args.group


def get_scores_path(group: str, metric: str) -> str:
    """Get scores path for the specified group."""
    scores_dir = "data/scores"
    if not os.path.isdir(scores_dir):
        os.makedirs(scores_dir)
    return f"{scores_dir}/{metric}_{group}_scores.txt"
