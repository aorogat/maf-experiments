from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any, ClassVar, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class DatasetProfileToolInput(BaseModel):
    """Input schema for DatasetProfileTool."""

    dataset_name: str = Field(
        default="",
        description="Dataset identifier. Supported values are EU-IT, WiFi, Utility, Volkert, and Yelp.",
    )


class DatasetProfileTool(BaseTool):
    name: str = "dataset_profile"
    default_dataset_name: str = "EU-IT"
    description: str = (
        "Returns factual profile information for a supported dataset, including shape, "
        "column types, missing-value counts, cardinality, numeric summary statistics, "
        "categorical top values, and target class distribution. The output contains "
        "facts only and no procedural advice. Supported datasets are EU-IT, WiFi, Utility, Volkert, and Yelp."
    )
    args_schema: Type[BaseModel] = DatasetProfileToolInput
    _call_history: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    DATASETS: ClassVar[dict[str, dict[str, str]]] = {
        "EU-IT": {
            "name": "EU-IT",
            "file": "EU-IT_cleaned.csv",
            "target_column": "Position",
        },
        "WiFi": {
            "name": "WiFi",
            "file": "WiFi.csv",
            "target_column": "TechCenter",
        },
        "Utility": {
            "name": "Utility",
            "file": "Utility.csv",
            "target_column": "CSRI",
        },
        "Volkert": {
            "name": "Volkert",
            "file": "volkert.csv",
            "target_column": "class",
        },
        "Yelp": {
            "name": "Yelp",
            "file": "Yelp_Merged.csv",
            "target_column": "stars",
        },
    }

    def _run(self, dataset_name: str = "") -> str:
        requested_name = dataset_name or self.default_dataset_name
        dataset_config = self._resolve_dataset_config(requested_name)
        canonical_name = dataset_config["name"]
        self._call_history.append(
            {"dataset_name": requested_name, "resolved_dataset_name": canonical_name}
        )

        dataset_path = Path.cwd() / dataset_config["file"]
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        target_column = dataset_config["target_column"]
        rows = self._read_rows(dataset_path)
        if not rows:
            raise ValueError(f"Dataset is empty: {dataset_path.name}")

        fieldnames = list(rows[0].keys())
        if target_column not in fieldnames:
            raise ValueError(f"Target column '{target_column}' was not found in {dataset_path.name}.")

        profile: dict[str, Any] = {
            "dataset_name": canonical_name,
            "source_file": dataset_path.name,
            "rows": len(rows),
            "columns": len(fieldnames),
            "target_column": target_column,
            "column_profiles": [],
        }

        for column_name in fieldnames:
            values = [row.get(column_name, "") for row in rows]
            present_values = [value for value in values if not self._is_missing(value)]
            numeric_values = [value for value in present_values if self._is_numeric(value)]
            is_numeric = bool(present_values) and len(numeric_values) == len(present_values)
            column_profile: dict[str, Any] = {
                "name": column_name,
                "dtype": "numeric" if is_numeric else "categorical",
                "missing_count": sum(1 for value in values if self._is_missing(value)),
                "cardinality": len({value for value in present_values}),
            }

            if is_numeric:
                clean_series = [float(value) for value in numeric_values]
                column_profile["numeric_summary"] = {
                    "min": min(clean_series) if clean_series else None,
                    "max": max(clean_series) if clean_series else None,
                    "mean": statistics.fmean(clean_series) if clean_series else None,
                    "std": statistics.stdev(clean_series) if len(clean_series) > 1 else None,
                }

            if not is_numeric:
                top_values = self._top_value_counts(values)
                column_profile["categorical_summary"] = {
                    "distinct_values": len({value for value in present_values}),
                    "top_values": top_values,
                }

            profile["column_profiles"].append(column_profile)

        target_counts = self._top_value_counts([row.get(target_column, "") for row in rows], limit=None)
        profile["target_distribution"] = {
            "classes": list(target_counts.keys()),
            "counts": target_counts,
        }

        return json.dumps(profile, indent=2)

    @classmethod
    def _resolve_dataset_config(cls, dataset_name: str) -> dict[str, str]:
        normalized_name = str(dataset_name).strip()
        aliases = {
            "wifi": "WiFi",
            "wi-fi": "WiFi",
            "eu-it": "EU-IT",
            "euit": "EU-IT",
            "utility": "Utility",
            "volkert": "Volkert",
            "yelp": "Yelp",
        }
        canonical_name = aliases.get(normalized_name.lower(), normalized_name)
        if canonical_name not in cls.DATASETS:
            supported = ", ".join(sorted(cls.DATASETS))
            raise ValueError(
                f"dataset_profile only supports these datasets: {supported}."
            )
        return cls.DATASETS[canonical_name]

    @staticmethod
    def _read_rows(dataset_path: Path) -> list[dict[str, str]]:
        with dataset_path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        return str(value).strip() == ""

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        if DatasetProfileTool._is_missing(value):
            return False
        try:
            float(str(value).strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def _top_value_counts(values: list[Any], limit: int | None = 5) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            key = "<<MISSING>>" if DatasetProfileTool._is_missing(value) else str(value).strip()
            counts[key] = counts.get(key, 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            sorted_counts = sorted_counts[:limit]
        return dict(sorted_counts)

    def get_call_history(self) -> list[dict[str, Any]]:
        return list(self._call_history)

    def reset_call_history(self) -> None:
        self._call_history.clear()
