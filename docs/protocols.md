# Evaluation protocols

GLPBench evaluates models under matched transductive and semi-inductive
link-prediction settings.

## Transductive evaluation

The transductive test set masks edges between nodes that remain visible in the
message-passing graph. Test-positive edges are excluded from that graph and
stored once as labeled candidate pairs.

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

## Feature preprocessing

Reducers are fitted only on training-visible nodes. The fitted transformation
is then applied to all nodes, including validation and test queries. Available
reducers include PCA, truncated SVD, and sparse random projection.
