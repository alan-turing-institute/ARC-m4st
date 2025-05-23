# Much of this file is taken from https://github.com/google-research/metricx
# It has been adapted to work as a standalone metric in the m4st package using the
# MetricXScore class. The original license is retained below.
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model classes for MetricX, modified from the T5 versions in HF."""

import copy
import dataclasses
import tempfile
import warnings

import datasets
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.mt5.modeling_mt5 import (
    __HEAD_MASK_WARNING_MSG,
    MT5Config,
    MT5PreTrainedModel,
    MT5Stack,
)

from m4st.metrics import Metric, TranslationDataset

METRICX_TOKENIZERS = {
    "google/metricx-24-hybrid-xxl-v2p6": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6": "google/mt5-large",
    "google/metricx-24-hybrid-xxl-v2p6-bfloat16": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6-bfloat16": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6-bfloat16": "google/mt5-large",
}


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 model for regression."""

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        decoder_head_mask: torch.FloatTensor | None = None,
        cross_attn_head_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.FloatTensor] | MT5ForRegressionOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask,
        # decoder_head_mask
        if (
            head_mask is not None
            and decoder_head_mask is None
            and self.config.num_layers == self.config.num_decoder_layers
        ):
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning, stacklevel=2)
            decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,  # type: ignore[misc]
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,  # type: ignore[misc]
            )

        hidden_states = encoder_outputs[0]  # type: ignore[index]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Create 1 step of dummy input for the decoder.
        batch_size = input_ids.size(0)  # type: ignore[union-attr]
        decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
        if torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.to(torch.device("cuda"))

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)  # type: ignore[union-attr]
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See
            # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # 250089 = <extra_id_10>
        predictions = lm_logits[:, 0, 250089]

        # Clip to 0 to 25
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # move labels to correct device to enable PP
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(
            loss=loss,
            predictions=predictions,
        )


class MetricXScore(Metric):
    """Applies MetricX 2024: https://github.com/google-research/metricx"""

    def __init__(
        self,
        model: str = "google/metricx-24-hybrid-large-v2p6",
        max_input_length: int = 1536,
        batch_size: int = 1,
        qe: bool = False,
        use_cpu: bool = False,
    ) -> None:
        if model not in METRICX_TOKENIZERS:
            msg = f"{model} is not a known MetricX model."
            raise KeyError(msg)

        self.tokenizer = AutoTokenizer.from_pretrained(METRICX_TOKENIZERS[model])
        self.model = MT5ForRegression.from_pretrained(model)
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.qe = qe
        self.use_cpu = use_cpu
        self.name = model.replace("/", "_")  # for output file paths

        self.data_req_inputs = ["prediction", "source"]
        self.metricx_req_inputs = ["hypothesis", "source"]
        if self.qe:
            self.name += "_qe"
        else:
            self.name += "_ref"
            self.data_req_inputs.append("reference")
            self.metricx_req_inputs.append("reference")

        self.field_mapping = dict(
            zip(self.metricx_req_inputs, self.data_req_inputs, strict=True)
        )

    def preprocess(self, hf_dataset: datasets.Dataset) -> datasets.Dataset:
        """Gets the test dataset for prediction.

        If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
        If it is false, there must be "hypothesis" and "reference" fields.

        Args:
            dataset: TranslationDataset instance

        Returns:
            Preprocessed and tokenized HuggingFace dataset compatible with MetricX.
        """

        def _make_input(example):
            if self.qe:
                example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                )
            else:
                example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example

        def _tokenize(example):
            return self.tokenizer(
                example["input"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        hf_dataset = hf_dataset.map(_make_input)
        hf_dataset = hf_dataset.map(_tokenize)
        hf_dataset = hf_dataset.map(_remove_eos)
        hf_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device="cpu" if self.use_cpu else None,
            output_all_columns=True,
        )
        return hf_dataset

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset)

        hf_dataset = datasets.Dataset.from_dict(dataset.to_dict(self.field_mapping))
        hf_dataset = self.preprocess(hf_dataset)

        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_eval_batch_size=self.batch_size,
                use_cpu=self.use_cpu,
            )
            trainer = Trainer(model=self.model, args=training_args)
            predictions, _, _ = trainer.predict(test_dataset=hf_dataset)

        return [float(pred) for pred in predictions]
