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
        blaser_lang_code_config=args["blaser_lang_config"],
        comet_model_str=args["comet_model"],
        metricx_model_str=args["metricx_model"],
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
            "COMET",
            "BLASER_ref",
            "BLASER_qe",
            "MetricX_ref",
            "MetricX_qe",
            "BLEU",
            "ChrF",
            "ChrF2",
        ],
        help="Metrics to use. Must be one or more \
            of COMET_ref, COMET_qe, BLASER_ref, BLASER_qe, MetricX_ref, MetricX_qe, \
            BLEU, ChrF, ChrF2. Defaults to all.",
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        type=int,
        required=False,
        help="Specific DEMETR disfluency \
            categories to be processed. By default all will be processsed.",
    )
    parser.add_argument(
        "--blaser-lang-config",
        type=str,
        default="../../configs/demetr/BLASER_lang_code_conversion.yaml",
        help="YAML config file mapping DEMETR language codes to SONAR language codes.",
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        required=False,
        help="COMET model to use.",
        default="Unbabel/wmt22-comet-da",
    )

    parser.add_argument(
        "--metricx-model",
        type=str,
        required=False,
        help="MetricX model to use.",
        default="google/metricx-24-hybrid-xl-v2p6",
    )

    args = parser.parse_args()
    main(vars(args))
