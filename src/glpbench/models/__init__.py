"""Link-prediction models, adapters, and capabilities."""

from glpbench.models.link_prediction import (
    AGAE_Model,
    DEALModel,
    FeatureCosineModel,
    FeatureMLPModel,
    GAE_Model,
    LinkPrediction,
    LinkPredictionModel,
    MLP_AGAE_Model,
    MLP_GAE_Model,
    ModelSpec,
    VGAE_Model,
    available_agae_regularizers,
    available_model_types,
    count_trainable_parameters,
    get_model_spec,
    sample_training_negative_edges,
)

__all__ = [
    "AGAE_Model",
    "DEALModel",
    "FeatureCosineModel",
    "FeatureMLPModel",
    "GAE_Model",
    "LinkPrediction",
    "LinkPredictionModel",
    "MLP_AGAE_Model",
    "MLP_GAE_Model",
    "ModelSpec",
    "VGAE_Model",
    "available_agae_regularizers",
    "available_model_types",
    "count_trainable_parameters",
    "get_model_spec",
    "sample_training_negative_edges",
]
