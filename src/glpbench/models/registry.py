"""Public registration and lookup for link-prediction model adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

from glpbench.contracts import ModelCapabilities
from glpbench.exceptions import CompatibilityError
from glpbench.interfaces import ModelAdapter
from glpbench.models.adapters import (
    AGAEAdapter,
    AGAEMLPAdapter,
    DEALAdapter,
    FeatureCosineAdapter,
    FeatureMLPAdapter,
    GAEAdapter,
    GAEMLPAdapter,
    VGAEAdapter,
)


_MODEL_ADAPTERS = {}
_NAME_LOOKUP = {}
_MODEL_SOURCES = {}
_LOADED_MODEL_FILES = {}


@dataclass(frozen=True, slots=True)
class ModelAdapterSource:
    """Describe the local file from which an adapter was loaded."""

    reference: str
    path: Path
    sha256: str


def normalize_model_name(name):
    """Return the case- and separator-insensitive form of a model name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    normalized = "".join(
        character for character in name.strip().lower()
        if character.isalnum()
    )
    if not normalized:
        raise ValueError("model name must contain an alphanumeric character")
    return normalized


def register_model_adapter(adapter, *, aliases=(), replace=False, _source=None):
    """Register a model adapter under its canonical name."""
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(adapter.capabilities, ModelCapabilities):
        raise TypeError("adapter capabilities must be a ModelCapabilities instance")
    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")

    canonical_name = adapter.name.strip().lower()
    canonical_key = normalize_model_name(adapter.name)
    lookup_keys = {
        canonical_key,
        *(normalize_model_name(alias) for alias in tuple(aliases)),
    }
    existing = _MODEL_ADAPTERS.get(canonical_name)
    if existing is not None and not replace:
        raise ValueError(f"model adapter '{canonical_name}' is already registered")
    if existing is None and canonical_key in _NAME_LOOKUP:
        raise ValueError(f"model name '{adapter.name}' collides with an existing alias")
    for key in lookup_keys:
        owner = _NAME_LOOKUP.get(key)
        if owner is not None and owner != canonical_name:
            raise ValueError(
                f"model alias collides with registered model '{owner}'"
            )

    adapters = dict(_MODEL_ADAPTERS)
    lookup = {
        key: owner for key, owner in _NAME_LOOKUP.items()
        if owner != canonical_name
    }
    adapters[canonical_name] = adapter
    lookup.update({key: canonical_name for key in lookup_keys})
    _MODEL_ADAPTERS.clear()
    _MODEL_ADAPTERS.update(adapters)
    _NAME_LOOKUP.clear()
    _NAME_LOOKUP.update(lookup)
    if _source is None:
        _MODEL_SOURCES.pop(canonical_name, None)
    else:
        _MODEL_SOURCES[canonical_name] = _source
    return adapter


def get_model_adapter(name):
    """Return the adapter registered for a model name or alias."""
    canonical_name = _NAME_LOOKUP.get(normalize_model_name(name))
    if canonical_name is None:
        choices = ", ".join(available_model_types())
        raise CompatibilityError(
            f"Unknown model '{name}'. Available models: {choices}.",
            component="model-registry",
            model=str(name),
        )
    return _MODEL_ADAPTERS[canonical_name]


def available_model_types():
    """Return registered canonical model names in stable order."""
    return tuple(sorted(_MODEL_ADAPTERS))


def get_model_adapter_source(name):
    """Return local-file provenance for a model, if applicable."""
    canonical_name = _NAME_LOOKUP.get(normalize_model_name(name))
    if canonical_name is None:
        get_model_adapter(name)
    return _MODEL_SOURCES.get(canonical_name)


def load_model_adapter_file(path, *, reference=None):
    """Load and register the single ``MODEL_ADAPTER`` exported by a file."""
    if not isinstance(path, (str, Path)):
        raise TypeError("model adapter path must be a string or Path")
    source_path = Path(path).expanduser().resolve()
    source_reference = str(path) if reference is None else str(reference)
    if source_path.suffix != ".py":
        raise CompatibilityError(
            "local model adapter must be a .py file",
            component="local-model-loader",
            context={"path": str(source_path), "stage": "path-validation"},
        )
    if not source_path.is_file():
        raise CompatibilityError(
            "local model adapter file does not exist",
            component="local-model-loader",
            context={"path": str(source_path), "stage": "path-validation"},
        )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    cached = _LOADED_MODEL_FILES.get(source_path)
    if cached is not None:
        cached_digest, canonical_name = cached
        if cached_digest != digest:
            raise CompatibilityError(
                "local model adapter file changed after it was loaded",
                component="local-model-loader",
                model=canonical_name,
                context={"path": str(source_path), "stage": "cache-validation"},
            )
        return _MODEL_ADAPTERS[canonical_name]

    adapters_before = dict(_MODEL_ADAPTERS)
    lookup_before = dict(_NAME_LOOKUP)
    sources_before = dict(_MODEL_SOURCES)
    module_name = f"_glpbench_local_model_{hashlib.sha256(str(source_path).encode()).hexdigest()}"
    parent = str(source_path.parent)
    added_parent = parent not in sys.path
    stage = "import"
    try:
        specification = importlib.util.spec_from_file_location(
            module_name, source_path
        )
        if specification is None or specification.loader is None:
            raise ImportError("could not create a module specification")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        if added_parent:
            sys.path.insert(0, parent)
        specification.loader.exec_module(module)
        stage = "export-validation"
        if _MODEL_ADAPTERS != adapters_before or _NAME_LOOKUP != lookup_before:
            raise ValueError(
                "local model files must export, not directly register, one adapter"
            )
        if not hasattr(module, "MODEL_ADAPTER"):
            raise AttributeError("file must export MODEL_ADAPTER")
        adapter = module.MODEL_ADAPTER
        source = ModelAdapterSource(source_reference, source_path, digest)
        stage = "registration"
        register_model_adapter(adapter, _source=source)
        canonical_name = adapter.name.strip().lower()
        _LOADED_MODEL_FILES[source_path] = (digest, canonical_name)
        return adapter
    except Exception as error:
        _MODEL_ADAPTERS.clear()
        _MODEL_ADAPTERS.update(adapters_before)
        _NAME_LOOKUP.clear()
        _NAME_LOOKUP.update(lookup_before)
        _MODEL_SOURCES.clear()
        _MODEL_SOURCES.update(sources_before)
        sys.modules.pop(module_name, None)
        if isinstance(error, CompatibilityError):
            raise
        raise CompatibilityError(
            f"failed to load local model adapter: {error}",
            component="local-model-loader",
            context={"path": str(source_path), "stage": stage},
        ) from error
    finally:
        if added_parent and parent in sys.path:
            sys.path.remove(parent)


def resolve_model_reference(reference, *, base_dir=None):
    """Resolve a registered name or load a local single-adapter file."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("model reference must be a non-empty string")
    candidate = Path(reference).expanduser()
    if candidate.suffix == ".py":
        if not candidate.is_absolute() and base_dir is not None:
            candidate = Path(base_dir) / candidate
        return load_model_adapter_file(
            candidate, reference=reference
        ).name.strip().lower()
    return get_model_adapter(reference).name.strip().lower()


def _register_builtins():
    for adapter_class in (
        GAEAdapter,
        GAEMLPAdapter,
        AGAEAdapter,
        AGAEMLPAdapter,
        VGAEAdapter,
        DEALAdapter,
        FeatureCosineAdapter,
        FeatureMLPAdapter,
    ):
        register_model_adapter(adapter_class(), replace=True)


_register_builtins()


__all__ = [
    "ModelAdapterSource",
    "available_model_types",
    "get_model_adapter",
    "get_model_adapter_source",
    "load_model_adapter_file",
    "normalize_model_name",
    "register_model_adapter",
    "resolve_model_reference",
]
