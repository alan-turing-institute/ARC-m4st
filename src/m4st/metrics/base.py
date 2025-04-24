import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Collection


@dataclasses.dataclass
class TranslationDataset:
    """
    Class to hold a translation dataset. Each field is a collection (list, tuple, etc.)
    of strings, plus an optional integer index field. The fields are:
    - prediction (required): the predicted (e.g. machine) translation
    - source (optional): Either the source text or a path to a .wav audio file to use as
        the source
    - reference (optional): the reference (e.g. human) translation
    - source_language (optional): the source language code
    - target_language (optional): the target language code
    - index (optional): an optional integer index for each sample
    """

    prediction: Collection[str]
    source: Collection[str] = ()
    reference: Collection[str] = ()
    source_language: Collection[str] = ()
    target_language: Collection[str] = ()
    index: Collection[int] = ()

    def __post_init__(self):
        # Check that all fields are the same length
        pred_length = len(self.prediction)
        for field in self.fields:
            if len(getattr(self, field)) != pred_length:
                msg = f"Field {field} has a different length to the prediction field."
                raise ValueError(msg)

    @property
    def fields(self) -> list[str]:
        """Return a list of all non-None fields in the dataset."""
        return [
            field.name
            for field in dataclasses.fields(self)
            if len(getattr(self, field.name)) > 0
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.prediction)

    def __iter__(self):
        """Iterate over the dataset, yielding a sample dict with a single value from
        each field."""
        for i in range(len(self)):
            yield {f: getattr(self, f)[i] for f in self.fields}

    def has_fields(self, fields: list[str]) -> bool:
        """Check the dataset has data for each in a list of fields."""
        return all(field in self.fields for field in fields)

    def has_audio_source(self) -> bool:
        if "source" not in self.fields:
            return False

        return any(src.endswith(".wav") for src in self.source)

    def to_samples(
        self, mapping: dict | None = None
    ) -> list[dict[str, Collection[str | int] | None]]:
        """
        Convert the dataset to a list of sample dicts with renamed keys based on the
        mapping.
        """
        if mapping is None:
            mapping = {field: field for field in self.fields}
        return [
            {new_name: sample[ds_field] for new_name, ds_field in mapping.items()}
            for sample in self
        ]

    def to_dict(
        self, mapping: dict | None = None
    ) -> dict[str, Collection[str | int] | None]:
        """
        Convert the dataset to a dict with renamed keys based on the mapping.
        """
        if mapping is None:
            mapping = {field: field for field in self.fields}
        return {
            new_name: getattr(self, ds_field) for new_name, ds_field in mapping.items()
        }


class Metric(ABC):
    data_req_inputs: list[str]
    name: str

    @abstractmethod
    def get_scores(self, data: TranslationDataset) -> list[float]:
        """Function to compute metric scores.

        Args:
            data: TranslationDataset instance.

        Returns:
            list of scores.
        """

    def check_dataset_compatible(
        self, dataset: TranslationDataset, audio_source: bool = False
    ) -> None:
        if not dataset.has_fields(self.data_req_inputs):
            msg = f"{self.name} requires dataset with {self.data_req_inputs} fields."
            raise ValueError(msg)
        if audio_source and not dataset.has_audio_source():
            msg = f"{self.name} requires audio source dataset."
            raise ValueError(msg)
        if not audio_source and dataset.has_audio_source():
            msg = f"{self.name} requires text source dataset."
            raise ValueError(msg)
