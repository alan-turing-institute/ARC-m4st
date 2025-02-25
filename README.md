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

# Compiling notes

`brew install typst`

`typst compile notes.typ`

## CallHome Dataset

Go [https://ca.talkbank.org/access/CallHome](here), select the conversation language, create account, then you can download the "media folder". There you can find the .cha files, which contain the transcriptions. You will need to pre-process this data in combination with the Callhome translations dataset, which includes part of the pre-processing scripts. The README under ./scripts/callhome of this repo contains more information.

To load the transcripts as an iterator, use the m4st.callhome.pipeline.CallhomePipeline class after pre-processing the data.

## Ollama

To use the Ollama client, which is one way to corrupt sentences randomly, you need to install [https://ollama.com](Ollama), and run the server.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-M4ST/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-M4ST/actions
<!-- prettier-ignore-end -->
