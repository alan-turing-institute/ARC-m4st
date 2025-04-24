# Metrics for Speech Translation (M4ST)

[![Actions Status][actions-badge]][actions-link]

Evaluation of metrics for Speech Translation

## Installation

From source:
```bash
git clone https://github.com/alan-turing-institute/ARC-M4ST
cd ARC-M4ST
python -m pip install .
```

## Usage

The code used to generate the experiment results for the main experiments with the WMT, DEMETR, and CallHome datasets is in the [scripts directory](https://github.com/alan-turing-institute/ARC-m4st/tree/main/scripts). It also contains a 4th exploratory experiment looking at the potential impact of transcription errors.

Metric classes for `ChrF`, `BLEU`, `COMET`, `MetricX`, and `BLASER-2.0` are defined in [src/m4st/metrics](https://github.com/alan-turing-institute/ARC-m4st/tree/main/src/m4st/metrics). All of them provide a `get_scores` method which takes an instance of a `TranslationDataset`, which takes lists of predictions, and optionally sources, references, and languages (see the docstring of the [`TranslationDataset`](https://github.com/alan-turing-institute/ARC-m4st/blob/main/src/m4st/metrics/base.py#L7) class).

## Compiling notes

`brew install typst`

`typst compile notes.typ`

## CallHome Dataset

Go [https://ca.talkbank.org/access/CallHome](here), select the conversation language, create account, then you can download the "media folder". There you can find the .cha files, which contain the transcriptions. You will need to pre-process this data in combination with the Callhome translations dataset, which includes part of the pre-processing scripts. The README under ./scripts/callhome of this repo contains more information.

To load the transcripts as an iterator, use the m4st.callhome.pipeline.CallhomePipeline class after pre-processing the data.

## Ollama

To use the Ollama client, which is one way to corrupt sentences randomly, you need to install [https://ollama.com](Ollama), and run the server.

## License

Distributed under the terms of the [MIT license](LICENSE). Some of the datasets used and metric code use different permissive licenses - see the relevant files/their documentation.


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-M4ST/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-M4ST/actions
<!-- prettier-ignore-end -->
