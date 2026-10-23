"""Link-prediction metrics and evaluation services."""

from glpbench.evaluation.components import (
    load_registered_dataset,
    realize_registered_split,
    validate_registered_compatibility,
)
from glpbench.evaluation.compatibility import (
    validate_model_protocol_compatibility,
)
from glpbench.evaluation.pipeline import (
    prepare_candidate_data,
    prepare_validation_data,
)

__all__ = [
    "load_registered_dataset",
    "prepare_candidate_data",
    "prepare_validation_data",
    "realize_registered_split",
    "validate_model_protocol_compatibility",
    "validate_registered_compatibility",
]
