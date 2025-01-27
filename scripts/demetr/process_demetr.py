import argparse
import os

from m4st.process_demetr import ProcessDEMETR


def main(args: dict) -> None:
    output_dir = args["output_dir"]
    output_file = args["output_file"]

    os.makedirs(output_dir, exist_ok=True)

    demetr = ProcessDEMETR(
        metrics_to_use=args["metrics"],
        output_filepath=os.path.join(output_dir, output_file),
        demetr_root=args["dataset_dir"],
    )

    print(args["cats"])
    demetr.process_demetr(cats_to_process=args["cats"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../../datasets/demetr",
        help="Root dataset \
            for DEMETR containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../outputs/demetr",
        help="Path to output directory. Will be created by script.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="demetr_results.csv",
        help="Name for output CSV file.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["COMET_ref", "COMET_qe", "BLASER_ref", "BLASER_qe", "SacreBLEU"],
        help="Metrics to use. Must be one or more \
            of COMET_ref, COMET_qe, BLASER_ref, BLASER_qe, SacreBLEU. Defaults to all.",
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        type=int,
        required=False,
        help="Specific DEMETR disfluency \
            categories to be processed. By default all will be processsed.",
    )

    args = parser.parse_args()
    main(vars(args))
