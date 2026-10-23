# Evaluation protocols

GLPBench evaluates models under matched transductive and semi-inductive
link-prediction settings.

Protocols can be constructed by registered name while the procedural split
functions remain available for compatibility:

```python
from glpbench.protocols import create_split_protocol

protocol = create_split_protocol(
    "matched-semi-inductive",
    query_size=50,
    validation_query_size=50,
    train_ratio=0.8,
)
splits = protocol.split(data, seed=42)
report = protocol.audit(data, splits)
```

Built-in names are `matched-semi-inductive`,
`matched-semi-inductive-validation-only`, `semi-inductive`, and
`transductive`.

## Transductive evaluation

The transductive test set masks edges between nodes that remain visible in the
message-passing graph. Test-positive edges are excluded from that graph and
stored once as labeled candidate pairs.

The standalone `transductive` protocol accepts separate validation and test
edge ratios whose sum must be less than one. PyG's `RandomLinkSplit` withholds
the positive edges. GLPBench samples validation and test negatives uniformly
from original-graph non-edges at a 1:1 ratio; training negatives remain
epoch-resampled by the training loop.

## Semi-inductive evaluation

The semi-inductive test set selects query nodes whose attributes remain
available while all incident edges are removed from the observed topology.
Test candidates connect observed source nodes to query targets. Validation
uses a separate, disjoint query population.

## Matched comparisons

The semi-inductive split is constructed first. GLPBench then matches the
transductive test-positive count and balances the eligible training-positive
pools so that the two evaluation settings use comparable training data.

`audit_comparable_splits()` validates topology equality, query roles,
positive and negative edge exclusion, reverse-orientation leakage, and paired
test-set invariants. The current protocol expects a simple undirected graph
represented with both edge orientations.

Validation-only, standalone semi-inductive, and standalone transductive
protocols provide equivalent protocol-specific audits. Audit failures raise
`ProtocolError` rather than being reduced to warnings.

Every built-in procedural constructor completes its audit before returning a
`SplitBundle`; registered built-in adapters inherit the same guarantee. An
invalid split therefore cannot proceed to feature preprocessing or model
construction through the standard execution paths. The explicit `audit()`
methods remain available for checking retained or externally modified bundles.

## Split metadata

Every built-in protocol returns a `SplitBundle` with schema-versioned
`ProtocolMetadata`. Its `parameters` describe the resolved construction and
query-selection modes, `requested_counts` records integer capacities requested
by the protocol, and `realized_counts` records the contents of the completed
PyG partitions. Query-node IDs and the visibility mask remain first-class
fields on `SplitBundle` rather than being duplicated in metadata.

Validated datasets and realized splits also receive deterministic SHA-256
fingerprints. Dataset fingerprints cover canonical graph structure, raw node
features, directedness, and the declared dataset version. Split fingerprints
cover the dataset fingerprint, protocol configuration and seed, query roles,
message-passing edges, and labeled evaluation candidates. They are independent
of tensor device and storage order. Node IDs remain meaningful, so renumbering
the graph changes its fingerprint. Epoch-resampled training negatives and
experiment-specific feature preprocessing are intentionally excluded.

Candidate positive and negative counts use the number of stored candidates.
Message-passing counts use stored directed entries, so an undirected edge
represented in both orientations contributes two entries. Protocol audits
recompute these values and the split fingerprint from the partitions and
reject inconsistent metadata.

## Feature preprocessing

Reducers are fitted only on training-visible nodes. The fitted transformation
is then applied to all nodes, including validation and test queries. Available
reducers include PCA, truncated SVD, and sparse random projection.

Preprocessing operations are registered by name and executed as an ordered
pipeline. Built-ins are `identity`, `normalize-features`, `pca`,
`truncated-svd`, and `sparse-random-projection`. Normalization or identity
produces the encoder input `x`; an optional terminal reducer produces
`x_reduced` from that same representation. The pipeline, rather than an
individual reducer, supplies the training-visible fit mask.

Experiment, benchmark, search, and convergence-audit execution resolve their
datasets, split protocols, and preprocessing operations through the component
registries. Existing CLI and YAML fields remain compatibility shorthands for
those registered components. Search and convergence audits resolve the
validation-only matched protocol, so their preparation boundary never receives
test candidates.
