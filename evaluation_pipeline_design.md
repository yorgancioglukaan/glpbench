# Evaluation Pipeline Design

## Status

This document defines the initial design for repackaging GLPBench as a
model-agnostic evaluation library for attributed-graph link prediction. It
defines the intended scope and behavioral contracts before the existing code
is reorganized. Interfaces remain provisional until implemented and tested.

## 1. Purpose

The library provides reproducible and leakage-safe evaluation of link
prediction models under transductive and semi-inductive settings. Its primary
responsibility is to ensure that models are trained, selected, and evaluated
under explicit and comparable protocols.

The library is not organized around any single architecture. AGAE, GAE,
GAE-MLP, AGAE-MLP, VGAE, DEAL, feature-cosine, and feature-MLP are included as
reference implementations and baselines.

The intended users are researchers who want to:

- compare link-prediction models on identical graph splits and candidate
  edges;
- evaluate cold-start performance when new-node attributes are available;
- add custom datasets, models, preprocessing methods, or metrics;
- run reproducible hyperparameter searches and benchmark grids; and
- inspect the complete configuration and provenance of a result.

## 2. Design principles

### Protocols are first-class objects

An evaluation protocol must explicitly describe how message-passing edges,
training examples, validation examples, test examples, query nodes, and
negative edges are constructed. Protocol behavior must not be hidden inside a
model or training loop.

### Models are evaluated through a common interface

The pipeline should depend on declared model capabilities rather than model
names. Architecture-specific behavior and parameters belong to model adapters.

### Validation and test data remain isolated

Hyperparameter search, checkpoint selection, and early stopping must not
inspect either test split. Test data are used only after all selection
decisions have been made.

### Preprocessing follows the visibility boundary

Any fitted preprocessing operation must use only nodes and information visible
during training. A fitted transformation may subsequently be applied to an
unseen node when the protocol makes that node's attributes available.

### Comparisons are paired whenever possible

Models evaluated within the same repetition should receive the same graph
splits, positive examples, and evaluation negatives. Transductive and
semi-inductive test sets should be derived from a common split foundation when
the stated protocol calls for a matched comparison.

### Runs are reproducible and auditable

Every reported result should be traceable to a resolved configuration, random
seed, dataset version, split fingerprint, software environment, and model
implementation.

### Extensibility does not require editing the runner

Adding a model, dataset, metric, reducer, or protocol should require
implementing and registering the corresponding interface, not adding branches
throughout the benchmark runner.

## 3. Terminology and evaluation settings

### Observed graph

The observed graph contains the nodes, attributes, and edges available before
the evaluation protocol withholds information. It is the input from which all
splits are derived.

### Message-passing graph

The message-passing graph is the topology supplied to a graph encoder for a
particular training or evaluation split. Candidate positive edges in that
split must not also appear in its message-passing graph.

### Candidate edge

A candidate edge is an ordered node pair to which the model assigns a score.
Positive and negative candidate edges are stored separately or accompanied by
binary labels.

### Visible node

A visible node may contribute its topology and attributes to training,
subject to the selected split and preprocessing protocol.

### Query node

A query node is a node held out to simulate a new node at validation or test
time. Its attributes are available when it is scored, but none of its incident
edges are visible to the model during training.

### Transductive link prediction

Both endpoints of a test candidate edge are present in the training-visible
graph. Test positives are withheld edges between visible nodes.

### Semi-inductive link prediction

Each evaluated candidate edge connects a training-visible source node to an
unseen query target node. Query-node attributes are available, while all edges
incident on validation and test query nodes are excluded from training.

This definition deliberately fixes the direction of semi-inductive candidate
edges: query nodes appear only as targets. Edges between two query nodes are
not valid candidates under this protocol.

### Fully inductive link prediction

Neither endpoint is observed during training, or evaluation occurs on a new
graph. This setting is outside the initial library scope but should remain
possible to add through the protocol interface.

## 4. Supported protocol families

### 4.1 Transductive protocol

The transductive protocol randomly withholds observed edges for validation and
test while retaining both endpoint nodes in the message-passing graph.

Required guarantees:

1. Positive candidate edges are absent from the corresponding message-passing
   graph.
2. Negative candidates are not positive edges in the original observed graph,
   in either orientation for an undirected graph.
3. Positive and negative candidates do not overlap.
4. Training, validation, and test positive sets are mutually disjoint.
5. The split is deterministic for a fixed graph, configuration, and seed.

### 4.2 Semi-inductive protocol

The semi-inductive protocol selects disjoint validation and test query-node
sets. Every edge incident on either query population is excluded from the
training topology. Validation and test positives connect visible sources to
their respective query targets.

Required guarantees:

1. Validation and test query sets are disjoint.
2. Query nodes have at least one eligible incident positive edge unless the
   protocol explicitly permits empty-query cases.
3. No edge incident on a validation or test query node appears in the training
   message-passing graph.
4. Query nodes appear only as targets in semi-inductive candidate edges.
5. Candidate edges never connect two query nodes.
6. Negative candidates are absent from the original observed graph in either
   orientation for an undirected graph.
7. Validation data are sufficient for checkpoint selection without exposing
   test queries or test edges.
8. Fitted preprocessing excludes both validation and test query nodes.
9. The split is deterministic for a fixed graph, configuration, and seed.

### 4.3 Matched transductive and semi-inductive evaluation

A matched protocol derives both test settings from one split foundation. It
should make training conditions comparable while respecting the semantic
difference between edge withholding and node withholding.

The semi-inductive split is constructed first because query-node incidence
constrains its positive-edge capacity. The transductive test set is then
matched to the realized number of semi-inductive test positives. Training
positive counts are matched by subsampling the larger eligible training pool.
The disjoint semi-inductive validation split has its own requested size and is
not required to match either test split.

If the requested query population or split size is infeasible, split
construction must fail with a descriptive error rather than silently reduce
the request. Split metadata records both requested and realized node and edge
counts.

The following are shared across models within one repetition:

- training and validation message-passing graphs;
- validation candidates used for selection;
- transductive test candidates;
- semi-inductive test candidates;
- fitted-preprocessing visibility masks; and
- all random seeds used to construct these artifacts.

Training negatives may be resampled during optimization and are not required
to be identical across architectures unless a future protocol explicitly
states otherwise.

## 5. Leakage model

The library treats the following as information leakage:

- retaining a labeled positive edge in the message-passing graph used to score
  that edge;
- sampling a known positive edge as a negative, including its reverse in an
  undirected graph;
- allowing validation or test query edges into the training topology;
- fitting a feature transformation on validation or test query nodes;
- using test performance for hyperparameter selection, checkpoint selection,
  early stopping, or training-policy selection; and
- generating different evaluation examples for models intended to be compared
  on a paired basis.

Protocol audits should fail with actionable messages. Audit failures must not
be converted into warnings during benchmark execution.

## 6. Core abstractions

The examples below illustrate responsibilities rather than committing to
exact class names or signatures.

### 6.1 Dataset provider

```python
class DatasetProvider(Protocol):
    name: str

    def load(self, root: Path, device: str) -> GraphDataset:
        ...
```

A graph dataset contains a standardized `torch_geometric.data.Data` object and
metadata including:

- canonical name and dataset family;
- source URL and citation;
- dataset or processing version;
- directedness;
- feature availability and dimensionality;
- node and edge counts;
- loader preprocessing; and
- a content or source fingerprint when available.

Dataset normalization is preprocessing and should not be permanently embedded
in the raw dataset loader.

### 6.2 Split protocol

```python
class SplitProtocol(Protocol):
    def split(self, data: Data, seed: int) -> SplitBundle:
        ...

    def audit(self, data: Data, splits: SplitBundle) -> AuditReport:
        ...
```

`SplitBundle` is an immutable description of training, validation, and test
partitions. It also records query masks, visibility masks, candidate labels,
protocol settings, and a deterministic fingerprint.

### 6.3 Preprocessor

```python
class FeaturePreprocessor(Protocol):
    def fit(self, x: Tensor, fit_mask: Tensor, seed: int | None = None):
        ...

    def transform(self, x: Tensor) -> Tensor:
        ...
```

Preprocessors declare whether they require fitting, support sparse inputs, and
use randomness. The pipeline owns the visibility mask passed to `fit`.

### 6.4 Model adapter

```python
class LinkPredictionModel(Protocol):
    capabilities: ModelCapabilities

    def fit(
        self,
        train_data: Data,
        validation_data: Data,
        training_policy: TrainingPolicy,
    ) -> TrainingResult:
        ...

    def score_edges(self, data: Data, edge_index: Tensor) -> Tensor:
        ...
```

`ModelCapabilities` should declare at least:

- whether the model is trainable;
- whether it uses a message-passing graph;
- whether it requires reduced features;
- whether it supports unseen sources, unseen targets, or both;
- whether it requires labels or other auxiliary data; and
- which devices it supports.

The runner uses capabilities to validate compatibility and prepare inputs. It
must not infer behavior from a model's registered name.

### 6.5 Training policy

A training policy controls optimization independently of model architecture:

- maximum epochs;
- optimizer and its parameters;
- checkpoint strategy and selection metric;
- early-stopping policy;
- learning-rate scheduler; and
- deterministic-execution settings.

The resulting `TrainingResult` records the selected checkpoint, best epoch,
epochs executed, final epoch, timing components, loss history, and model
parameter count.

### 6.6 Metric

```python
class EvaluationMetric(Protocol):
    name: str

    def compute(self, labels: Tensor, scores: Tensor) -> float:
        ...
```

Metrics declare whether they consume raw scores, probabilities, rankings, or
thresholded predictions. ROC-AUC and average precision remain the initial
defaults. Ranking metrics may be added once their candidate-set semantics are
defined explicitly.

### 6.7 Result store

A result store persists structured run records without coupling evaluation to
CSV. The initial implementation may retain CSV while exposing a storage
interface that can later support Parquet or a database.

## 7. Experiment lifecycle

A complete experiment follows these stages:

1. Parse and validate the user configuration.
2. Resolve defaults into an immutable experiment specification.
3. Load raw dataset data and metadata.
4. Generate a typed, deterministic in-memory split bundle.
5. Audit all protocol invariants.
6. Fit preprocessing using the training-visible mask.
7. Construct the model through its registered adapter.
8. Train using only training and validation data.
9. Restore the checkpoint selected by validation performance.
10. Evaluate once on each requested test setting.
11. Record metrics, efficiency measurements, provenance, and failures.
12. Aggregate repeated seeds without discarding individual results.

Search execution stops after stage 8 and records validation results only. It
must not materialize test data through its public interface.

## 8. Configuration model

Configuration files must be versioned and validated before data loading or
training begins. Common settings and component-specific settings should be
namespaced.

```yaml
schema_version: 1

datasets:
  - cora
  - wikics

protocol:
  name: matched-transductive-semi-inductive
  parameters:
    query_fraction: 0.10
    validation_query_fraction: 0.10
    train_ratio: 0.80
    negative_ratio: 1.0

preprocessing:
  normalize_features: true
  reducer:
    name: pca
    parameters:
      output_dimensions: 64

models:
  - name: agae
    parameters:
      hidden_channels: 128
      regularization:
        name: cosine-alignment
        strength: 0.01
  - name: gae
    parameters:
      hidden_channels: 128

training:
  maximum_epochs: 500
  optimizer:
    name: adam
    parameters:
      learning_rate: 0.01
      weight_decay: 0.0
  checkpoint:
    strategy: best-validation
    metric: roc_auc

evaluation:
  metrics:
    - roc_auc
    - average_precision
  seeds: [101, 102, 103]

output:
  directory: out/example
```

Unknown keys should be rejected by default. The fully resolved configuration
must be written to the run directory.

## 9. Split bundles and fingerprints

The initial library represents each split as an immutable in-memory bundle.
The bundle contains the message-passing edges, fixed evaluation candidates,
query and visibility masks, protocol settings, audit results, dataset identity,
random seed, and a deterministic fingerprint. Training negatives are sampled
by the training loop and are not members of the bundle.

Persistent export and loading of split tensors are deferred. Core interfaces
must avoid assumptions that would prevent a safe, versioned artifact format
from being added later.

## 10. Outputs and provenance

The canonical output is a run directory rather than a single CSV file:

```text
runs/<run-id>/
    resolved_config.yaml
    environment.json
    results.csv
    epoch_history.csv
    failures.jsonl
    split_manifest.json
    summary.json
```

Each result should record:

- run, configuration, repetition, and split identifiers;
- dataset, protocol, model, preprocessing, and training settings;
- random seeds by purpose;
- validation and test metrics;
- requested epochs, epochs executed, and best epoch;
- training, preprocessing, and evaluation time;
- trainable parameter count and optional peak memory;
- package version and source revision; and
- dataset and split fingerprints.

Aggregates must report the number of successful repetitions and retain the
underlying per-seed rows. Failed jobs should be recorded without being treated
as missing-at-random observations.

## 11. Registries and third-party extensions

The initial release should expose registration functions for:

- dataset providers;
- split protocols;
- feature preprocessors;
- model adapters; and
- evaluation metrics.

Registration must reject duplicate names unless replacement is explicitly
requested. Built-in components and third-party components should follow the
same interface. A future release may support Python entry points for extensions
distributed as separate packages.

At least one tutorial must demonstrate integrating an external model without
editing the benchmark runner.

## 12. Errors and failure policy

Errors should identify the failing component, dataset, model, seed, and
configuration identifier. Expected validation failures should use specific
exception types rather than generic `ValueError` where practical.

Benchmark execution supports two policies:

- continue after recording an individual job failure; or
- fail immediately when `fail_fast` is enabled.

Resume mode must use stable identifiers derived from the resolved experiment
specification, not row position or execution order.

## 13. Testing requirements

### Unit tests

Test dataset standardization, samplers, reducers, metrics, registries,
configuration validation, training policies, and result serialization in
isolation.

### Protocol-invariant tests

For both hand-constructed and randomly generated graphs, verify every invariant
listed in Section 4. Important edge cases include:

- isolated nodes;
- complete and nearly complete graphs;
- graphs with insufficient negative-edge capacity;
- duplicate or bidirectional input edges;
- very small query populations;
- edges connecting two prospective query nodes;
- high-degree query nodes; and
- impossible requested split sizes.

### Determinism tests

A fixed graph, resolved configuration, and seed must reproduce query selection,
edge partitions, evaluation negatives, and fingerprints.

### Integration tests

Run a complete benchmark on a small synthetic graph with at least one trainable
and one non-trainable model. Integration tests must not download data.

### Regression tests

Use small deterministic fixtures to detect unintended changes in split
semantics, configuration expansion, logging schemas, and resume behavior.

### Extension tests

Provide minimal external dataset and model implementations and verify that they
can be registered and evaluated without changing core code.

## 14. Documentation requirements

The user documentation should contain:

1. Installation and a five-minute example.
2. Formal definitions of supported evaluation settings.
3. Protocol guarantees and known limitations.
4. CLI and configuration reference.
5. Python API reference.
6. Tutorials for adding a dataset, model, reducer, metric, and protocol.
7. Reproduction instructions for published benchmarks.
8. Dataset sources, licenses, citations, and preprocessing notes.
9. Versioning and backward-compatibility policy.

AGAE should be documented alongside the other included models rather than as
the package's primary abstraction.

## 15. Initial non-goals

The first library release does not need to support:

- fully inductive evaluation on entirely unseen graphs;
- temporal or heterogeneous graphs;
- distributed multi-GPU training;
- large-scale sampled message passing;
- dynamic graphs;
- exhaustive implementations of published link-prediction methods; or
- standardized hard-negative evaluation before its candidate semantics are
  formally specified.

These should not be precluded by core interfaces, but implementing them is not
required for the initial publication.

## 16. Publication validation

The accompanying evaluation should study the framework rather than promote a
single included model. Candidate research questions include:

1. How much do model rankings change between transductive and semi-inductive
   protocols?
2. How strongly do leakage-prone preprocessing and negative sampling inflate
   reported performance?
3. Does paired evaluation reduce variance when comparing architectures?
4. How sensitive are conclusions to checkpoint selection, early stopping, and
   fixed training budgets?
5. Can independently implemented models reproduce results when given the same
   resolved protocol configuration, dataset fingerprint, and seeds?
6. What runtime and storage overhead does protocol enforcement introduce?

Any publication claim should be linked to a specific experiment and acceptance
criterion before the experiment is run.

## 17. Proposed package layout

```text
pyproject.toml
src/
    glpbench/
        __init__.py
        cli.py
        configuration/
        datasets/
        evaluation/
        logging/
        models/
        preprocessing/
        protocols/
        runners/
        training/
tests/
    unit/
    integration/
    regression/
examples/
configs/
docs/
```

The package name, command name, and documentation title use the GLPBench
identity.

## 18. Migration plan

The refactor should preserve current behavior in small, testable stages:

1. Lock down current split and evaluation behavior with regression tests.
2. Add package metadata and create the `src` package structure.
3. Move existing modules into the package without changing logic.
4. Introduce typed experiment specifications and versioned configuration
   validation.
5. Extract dataset and preprocessing interfaces.
6. Extract split protocols and immutable split bundles.
7. Introduce the model-adapter and capability interfaces.
8. Separate training policies, evaluation, and result persistence.
9. Add graph and split fingerprints while keeping split bundles in memory.
10. Divide tests into unit, integration, regression, and invariant suites.
11. Replace old entry scripts with one package CLI and remove the temporary
    compatibility commands before the next alpha release.
12. Rewrite documentation around evaluation protocols and extension workflows.

Each stage should leave the test suite passing. Because the initial alpha had
no external users, its temporary root-level compatibility wrappers were
removed immediately rather than carrying them through a deprecation period.

## 19. Design decisions

### Resolved

- The public project name is **GLPBench**, the Python package is `glpbench`,
  and command-line entry points use the `glpbench` prefix.
- Matched evaluation constructs the semi-inductive split first, matches the
  transductive test-positive count to it, and matches training-positive counts
  by subsampling the larger eligible pool. Validation is a separate disjoint
  semi-inductive split. Infeasible requests raise an error, and requested and
  realized counts are recorded in split metadata.
- Undirected-edge handling preserves the behavior of the existing framework.
  Source graphs and message-passing graphs are represented bidirectionally,
  while each labeled candidate pair is stored once. Membership, overlap, and
  negative-edge checks treat both orientations as the same underlying edge.
  Decoders score the ordered pair supplied in `edge_label_index` without
  averaging or otherwise combining reverse-orientation scores. Semi-inductive
  candidates retain visible-source/query-target ordering. Any change to score
  symmetrization is deferred to a separately specified and tested protocol
  revision rather than being included in the library refactor.
- Reusable split artifacts do not contain training negatives. The training
  loop continues to sample one negative per training positive at every epoch,
  using the existing deterministic seed schedule of `run_seed + epoch`.
  Artifacts contain the fixed validation and test candidates, while the
  resolved run configuration records the training sampler and seed. This
  preserves current training behavior without coupling an artifact to an
  epoch budget.
- Persistent split artifacts are deferred beyond the initial library
  refactor. Splits remain typed in-memory objects generated deterministically
  from the dataset, resolved protocol configuration, and seed. Their interface
  should not preclude later serialization, but the initial refactor adds no
  artifact format or serialization dependency.
- CSV remains the canonical public result format for the initial library.
  Internally, runners produce typed `EvaluationResult` objects that are passed
  to a persistence interface; the initial implementation retains the existing
  CSV backend. YAML remains the experiment-specification format and JSON Lines
  remains the failure-log format. Additional result backends such as Parquet
  or SQLite are deferred.
- Configuration loading continues to use explicit parsing and validation with
  standard-library dataclasses. Resolved configurations are typed and
  immutable, unknown fields are rejected, and validation errors identify the
  relevant configuration path. Top-level formats are schema-versioned. A
  third-party validation dependency and automatic schema generation are
  deferred until configuration complexity justifies them.
- Official support initially covers Python 3.10--3.12, PyTorch 2.1 up to but
  excluding 3.0, PyG 2.5 up to but excluding 3.0, and CPU execution. CUDA and
  Apple MPS are best-effort until covered by dedicated tests. A version is
  considered supported only when exercised by the automated GitHub Actions
  test workflow. Exact Python, PyTorch, PyG, and device information is recorded
  with benchmark provenance. Python 3.13 must not be advertised until it is
  included in that workflow.
- A user-provided dataset supplies a stable name, a version or revision, its
  directedness, and a PyG `Data` object with a two-dimensional numeric `x`, an
  integer `edge_index` of shape `[2, E]`, and an explicit or unambiguously
  inferable node count. GLPBench computes structural summaries and graph and
  feature fingerprints. Source URL, citation, license, and description are
  optional. Initial protocols require attributed simple undirected graphs;
  incompatible inputs raise descriptive errors rather than being silently
  repaired.
- Model--protocol compatibility is determined declaratively. Protocols state
  their endpoint-visibility, input, ordering, graph, and evaluation
  requirements; model adapters state their capabilities, required inputs,
  preprocessing outputs, and supported devices. The runner checks that the
  model capabilities and dataset inputs satisfy the protocol before
  preprocessing or training. Compatibility must not be inferred from a model
  name. Built-in adapters have contract tests for every claimed protocol, and
  reusable conformance tests are provided for third-party adapters.

### Open

No design decisions currently remain open for the initial refactor.

## 20. Implementation to-do list

The migration must preserve current numerical and protocol behavior unless a
separate change is explicitly proposed, tested, and documented. Each phase
must leave the full test suite passing. Priorities below reflect the order in
which work should begin, not the conceptual order of the preceding sections.

### P0: Establish the library migration boundary

- [x] Add regression fixtures for the current split outputs, configuration
  expansion, per-epoch training-negative seed schedule, result columns, and
  CLI dispatch behavior.
- [ ] Add a normal GitHub Actions test workflow for pushes and pull requests
  on Python 3.10, 3.11, and 3.12 using CPU execution.
- [ ] Align package metadata with the support policy: remove the untested
  Python 3.13 classifier and declare PyTorch and PyG major-version limits.
- [x] Adopt a `src/glpbench/` package layout and configure setuptools for it.
  Setuptools now discovers packages automatically under `src`.
- [x] Move implementation modules into `src/glpbench/` in small batches while
  preserving behavior.
- [x] Replace dynamic imports of root modules in `glpbench.cli` with package
  imports and verify that the installed CLI does not depend on the repository
  working directory.
- [x] Move the regression suite into a discoverable top-level `tests/`
  directory without reducing coverage.
- [x] Divide the monolithic suite into `tests/unit`, `tests/integration`,
  `tests/regression`, and `tests/invariants` without changing test behavior.
- [x] Remove root-level modules and legacy entry scripts before the next alpha
  release; no external users depended on the temporary compatibility surface.

Completion criterion: a clean wheel contains the implementation under
`glpbench`, package-native imports and CLI commands work outside the source
checkout, and the supported Python matrix passes automatically.

### P1: Introduce typed contracts without changing execution

- [ ] Add immutable dataclasses for `DatasetMetadata`, `GraphDataset`,
  `SplitBundle`, `ProtocolMetadata`, `ModelCapabilities`, `TrainingPolicy`,
  `TrainingResult`, `EvaluationResult`, and resolved experiment
  specifications.
- [ ] Add protocol interfaces for dataset providers, split protocols,
  preprocessors, model adapters, metrics, and result stores.
- [ ] Add component-specific exception types for configuration, dataset,
  protocol, compatibility, training, and persistence failures.
- [ ] Wrap existing return values in the typed contracts at module boundaries;
  keep the underlying algorithms unchanged.
- [ ] Add strict, schema-versioned YAML parsing into frozen dataclasses, reject
  unknown keys, and include full field paths in validation errors.
- [ ] Preserve support for current benchmark, search, ablation, and audit
  configuration files through compatibility parsers or a documented one-time
  schema conversion.

Completion criterion: runners exchange typed objects internally while all
existing configurations and regression fixtures retain their behavior.

### P2: Make protocols and data preparation first-class

- [ ] Implement dataset-provider registration and adapt every built-in loader
  to return `GraphDataset` with the required metadata.
- [ ] Add validation for user-provided attributed simple undirected graphs,
  including tensor shapes, dtypes, node bounds, symmetry, duplicates, and
  self-loops; do not silently repair invalid data.
- [ ] Extract transductive, semi-inductive, and matched split construction into
  registered `SplitProtocol` implementations backed by the existing split
  functions.
- [ ] Encode requested and realized split counts, query and visibility masks,
  evaluation candidates, and protocol parameters in `SplitBundle`.
- [ ] Enforce the finalized matched-capacity rule and provide descriptive
  errors for infeasible requests.
- [ ] Move leakage and split-invariant checks behind each protocol's `audit`
  method and require a successful audit before model construction.
- [ ] Add deterministic dataset and split fingerprints without serializing
  split tensors.
- [ ] Adapt normalization, PCA, truncated SVD, and sparse random projection to
  the preprocessor interface, with the pipeline controlling every fit mask.
- [ ] Add randomized invariant tests and edge cases for isolated nodes,
  near-complete graphs, duplicate input edges, small query populations,
  query--query edges, and insufficient split or negative capacity.

Completion criterion: runners request a protocol by registration name and no
longer call split or preprocessing implementations directly.

### P3: Introduce model adapters and declarative compatibility

- [ ] Define protocol requirements and implement the capability-matching
  function with actionable incompatibility errors.
- [ ] Create adapters for all bundled models: GAE, GAE-MLP, AGAE, AGAE-MLP,
  VGAE, DEAL, feature-cosine, and feature-MLP.
- [ ] Move architecture-specific parameter handling, preprocessing needs, and
  training behavior into adapters without changing score semantics.
- [ ] Remove model-name conditionals from shared runners once equivalent
  adapter behavior is covered by regression tests.
- [ ] Implement model registration with duplicate-name rejection and explicit
  opt-in replacement.
- [ ] Provide reusable conformance tests and run them against every built-in
  adapter and a minimal third-party example adapter.

Completion criterion: adding a conforming model requires registration and an
adapter, with no edits to benchmark, search, or experiment runners.

### P4: Separate orchestration, training, evaluation, and persistence

- [ ] Extract the training loop into a service driven by `TrainingPolicy`,
  preserving checkpoint selection, early stopping, scheduling, timing, and
  the `run_seed + epoch` negative-resampling schedule.
- [ ] Extract metrics into registered implementations that declare the score
  representation they consume.
- [ ] Create one experiment service shared by single runs, benchmarks,
  searches, and convergence audits.
- [ ] Ensure search execution exposes validation results only and cannot
  access test candidates through its public interface.
- [ ] Make runners emit typed `EvaluationResult` records rather than writing
  CSV fields directly.
- [ ] Introduce a result-store interface and adapt the existing CSV logger as
  the canonical backend; retain JSONL failure logs.
- [ ] Preserve stable run and job identifiers, resume behavior, paired splits,
  individual seed rows, and current failure policies.
- [ ] Record resolved configuration, exact software versions, device, dataset
  fingerprint, split fingerprint, timing, epoch counts, and parameter counts.
- [ ] Add download-free end-to-end tests on a synthetic graph with one
  trainable and one non-trainable model.

Completion criterion: all runners use the same orchestration path, differ only
in job generation and permitted outputs, and reproduce the existing logged
results for fixed fixtures.

### P5: Stabilize the public API and documentation

- [ ] Define and export the intentionally supported API from
  `glpbench.__init__`; keep internal helpers private.
- [ ] Review public docstrings, type annotations, exception contracts, and
  reproducibility guarantees.
- [x] Consolidate the CLI under `glpbench` subcommands and remove temporary
  legacy commands before the next alpha release.
- [ ] Add a five-minute installed-package example that uses no repository-local
  imports.
- [ ] Document protocol definitions, leakage guarantees, configuration and CLI
  references, dataset provenance, and backward-compatibility policy.
- [ ] Add tutorials for registering a custom dataset, model, preprocessor,
  metric, and split protocol without modifying core runner code.
- [ ] Add a release checklist covering clean-environment installation, wheel
  contents, test matrix, documentation links, and PyPI trusted publishing.

Completion criterion: a new user can install GLPBench, run a small experiment,
and extend one component using only documented public interfaces.

### Deferred beyond the initial refactor

- [ ] Persistent split export and safe tensor serialization.
- [ ] Additional result backends such as Parquet or SQLite.
- [ ] Python 3.13, CUDA, and Apple MPS support claims until covered by CI.
- [ ] Python entry-point discovery for extensions in separately distributed
  packages.
- [ ] Fully inductive, temporal, heterogeneous, dynamic, distributed, and
  sampled-message-passing protocols.
- [ ] Hard-negative and ranking-metric protocols until their candidate-set
  semantics are specified.
