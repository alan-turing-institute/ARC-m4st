import os

import pandas as pd
from tqdm import tqdm


def mapping_name_to_filename(mname):
    return mname[3:] + ".cha"


def process_line_ids(maybe_lines):
    return [int(num) for num in maybe_lines.split("_")]


def reconcile_timestamps(timestamps: list[str]):
    r"""
    Returns the maximum overlapping timestamp of the input set.
    """
    min_min: None | int = None
    max_max: None | int = None
    for ts in timestamps:
        _min = int(ts.split("_")[0])
        _max = int(ts.split("_")[1])
        if min_min is None or _min < min_min:
            min_min = _min
        if max_max is None or _max > max_max:
            max_max = _max

    assert min_min is not None
    assert max_max is not None
    return "_".join([str(min_min), str(max_max)])


def process(cha_root_dir, mapping_file, eng_file) -> pd.DataFrame:
    r"""
    Loads a DataFrame of Spanish utterances alongside their English translations.

    This function expects that the dataset has been pre-processed, so that the
    lines in the mapping file match to the correct lines in the transcription files.
    """
    with open(eng_file) as eng_in:
        eng_lines = [line[:-1] for line in eng_in.readlines()]

    lines = []
    fnames = []
    with open(mapping_file) as mapping_in:
        for fid_line in mapping_in.readlines():
            cha_prefix, line = fid_line.split(" ")
            fname = mapping_name_to_filename(cha_prefix)
            line = line[:-1]

            lines.append(line)
            fnames.append(fname)

    spa_file_to_lines: dict[str, list[str]] = {}
    spa_file_to_timestamps: dict[str, list[str]] = {}
    spa_eng_dict: dict[str, list[str]] = {
        "prefix": [],
        "spa": [],
        "eng": [],
        "timestamp": [],
    }
    for idx, fname in tqdm(enumerate(fnames)):
        # Following block ensures we have loaded the spanish lines and timestamps
        if fname not in spa_file_to_lines:
            # Load the Spanish text, and corresponding timestamps.
            with open(os.path.join(cha_root_dir, fname)) as spa_in:
                # Filter out metadata lines and load in the rest.
                spa_file_to_lines[fname] = [
                    line for line in spa_in.readlines() if line[0] != "@"
                ]
            with open(
                os.path.join(cha_root_dir, fname + "_timestamps.txt")
            ) as spa_ts_in:
                spa_file_to_timestamps[fname] = spa_ts_in.readlines()

        all_spa_lines = spa_file_to_lines[fname]
        all_spa_timestamps = spa_file_to_timestamps[fname]
        spa_line_num = lines[idx]
        maybe_spa_line_nums = process_line_ids(
            spa_line_num
        )  # Possibly inputting a formatted string

        # The English translation contains more text than in the original Spanish
        # files, so the line numbers might exceed the length of the file.
        # In this case, skip past the line.
        skip_line = False
        for spa_line_num in maybe_spa_line_nums:
            if spa_line_num >= len(spa_file_to_lines[fname]):
                skip_line = True

        if skip_line:
            continue

        spa_line_final = " ".join(
            [all_spa_lines[num - 1] for num in maybe_spa_line_nums]
        ).rstrip("\n")
        all_timestamps = [all_spa_timestamps[_idx] for _idx in maybe_spa_line_nums]
        # Return the smallest timestamp that overlaps with all given timestamps.
        out_timestamp = reconcile_timestamps(all_timestamps).rstrip("\n")

        spa_eng_dict["spa"].append(spa_line_final)
        spa_eng_dict["eng"].append(eng_lines[idx])
        # NOTE That you can use the multiple timestamps to separate cases of people
        # talking over each other, hence can carry out test against.
        spa_eng_dict["timestamp"].append(out_timestamp)
        spa_eng_dict["prefix"].append(fname)

    return pd.DataFrame.from_dict(spa_eng_dict)


if __name__ == "__main__":
    r"""
    Pairs English callhome translations with Spanish dialogue lines,
    ignoring extra English content.
    """
    cha_root_dir = (
        "/Users/bvodenicharski/repos/ARC-m4st/experiments/compare_text_content/spatext"
    )
    mapping_root = (
        "/Users/bvodenicharski/repos/ARC-m4st/data/fisher_ch_spa-eng/data/mapping"
    )
    eng_file_root = (
        "/Users/bvodenicharski/repos/ARC-m4st/data/fisher_ch_spa-eng/data/corpus/ldc"
    )

    for callhome_partition in [
        "callhome_devtest",
        "callhome_evltest",
        "callhome_train",
    ]:
        print(f"Processing {callhome_partition}")
        mapping_file = os.path.join(mapping_root, callhome_partition)
        eng_file = os.path.join(eng_file_root, callhome_partition + ".en")
        process(cha_root_dir, mapping_file, eng_file)
