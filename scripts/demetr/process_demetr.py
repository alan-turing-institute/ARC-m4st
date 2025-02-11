import argparse
import os

from m4st.process_demetr import ProcessDEMETR


def main(args: dict) -> None:
    output_dir = args["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    demetr = ProcessDEMETR(
        metrics_to_use=args["metrics"],
        output_dir=output_dir,
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
        "--metrics",
        nargs="+",
        type=str,
        default=[
            "COMET_ref",
            "COMET_qe",
            "BLASER_ref",
            "BLASER_qe",
            "Bleu",
            "ChrF",
            "ChrF2",
        ],
        help="Metrics to use. Must be one or more \
            of COMET_ref, COMET_qe, BLASER_ref, BLASER_qe, Bleu, ChrF, ChrF2. \
            Defaults to all.",
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
