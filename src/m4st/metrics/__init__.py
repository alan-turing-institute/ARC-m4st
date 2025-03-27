import os
from abc import ABC, abstractmethod

from pandas import DataFrame


class Metric(ABC):
    @abstractmethod
    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        """Function definition for all metrics. Assumes use of the DEMETR dataset.

        cat_data --     pd.DataFrame object created by directly loading a DEMETR JSON
                        file (e.g. base_id35_reference.json) with pd.read_json.
        output_path --  Directory for storing output JSON files. There will be one
                        output file for each DEMETR category, for each metric.
        input_fp --     Path to input JSON file from the DEMETR dataset.
        ghfghgfhj
        """
