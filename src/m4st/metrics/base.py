import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Collection


@dataclasses.dataclass
class TranslationDataset:
    prediction: Collection[str]
    source: Collection[str] = ()
    reference: Collection[str] = ()
    source_language: Collection[str] = ()
    index: Collection[int] = ()

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
        """Iterate over the dataset, yielding a single sample from each field."""
        for i in range(len(self)):
            yield {f: getattr(self, f)[i] for f in self.fields}

    def has_fields(self, fields: list[str]) -> bool:
        """Check the dataset has data for each in a list of fields."""
        return all(field in self.fields for field in fields)

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

    def check_dataset_compatible(self, dataset: TranslationDataset) -> None:
        if not dataset.has_fields(self.data_req_inputs):
            msg = f"{self.name} requires dataset with {self.data_req_inputs} fields."
            raise ValueError(msg)
