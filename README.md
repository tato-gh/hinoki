# Hinoki

LightGBM bindings for Elixir.

Hinoki provides training and prediction for `Nx.Tensor` and
`Explorer.DataFrame` inputs, model save/load helpers, early stopping,
feature importance, permutation importance, k-fold cross-validation, and
grid search.

## Usage

Tensor input:

```elixir
{features, labels} = some_data()

booster =
  Hinoki.train({features, labels},
    num_iterations: 100,
    params: [objective: "binary", num_threads: 4, learning_rate: 0.05, seed: 42]
  )

preds = Hinoki.predict(booster, features)
```

DataFrame input:

```elixir
booster =
  Hinoki.train(df,
    target: :label,
    params: [objective: "regression", num_threads: 1, seed: 42]
  )

preds = Hinoki.predict(booster, Explorer.DataFrame.discard(df, ["label"]))
```

Parameters are forwarded to LightGBM. Set `num_threads` explicitly,
because LightGBM uses OpenMP internally and can otherwise consume every
CPU on the machine.

DataFrame columns with Explorer's `:category` dtype are passed to
LightGBM as categorical features automatically. Tensor input has no
column dtype metadata, so pass `categorical_feature` in `:params` when
using `Nx.Tensor` features.

## Introspection and Importance

```elixir
gain = Hinoki.feature_importance(booster)
split = Hinoki.feature_importance(booster, :split)
named_gain = Hinoki.named_feature_importance(booster, [:x1, :x2])

Hinoki.num_features(booster)
Hinoki.num_classes(booster)
Hinoki.current_iteration(booster)
Hinoki.categorical_features(booster)
Hinoki.info(booster, :params)
Hinoki.version()

Hinoki.permutation_importance(booster, features, labels, fn y, pred ->
  Nx.mean(Nx.pow(Nx.subtract(y, pred), 2))
end,
  features: [0, 1],
  n_repeats: 5,
  seed: 42
)
```

## Early Stopping

Early stopping uses validation data and records the best validation
result in the returned booster.

```elixir
booster =
  Hinoki.train({train_features, train_labels},
    valid: {valid_features, valid_labels},
    early_stopping_rounds: 10,
    num_iterations: 500,
    params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42]
  )

Hinoki.current_iteration(booster)
Hinoki.best(booster)
```

## Ranking Groups

Ranking objectives such as `lambdarank` require explicit query/group
metadata. For tensor input, pass group sizes whose sum equals the row
count. Validation data uses `:valid_group`.

```elixir
booster =
  Hinoki.train({features, labels},
    group: [16, 18, 12],
    valid: {valid_features, valid_labels},
    valid_group: [15, 17],
    params: [objective: "lambdarank", metric: "ndcg", num_threads: 1, seed: 42]
  )
```

For DataFrame input, pass the group column name. Hinoki removes both the
target and group columns from the feature matrix, converts contiguous
group values to group sizes, and sets LightGBM's `group` dataset field.
The DataFrame must already be ordered so each group is one contiguous
block. Validation DataFrames use the same group column unless
`:valid_group` names a different one.

```elixir
booster =
  Hinoki.train(df,
    target: :label,
    group: :group_label,
    params: [objective: "lambdarank", metric: "ndcg", num_threads: 1, seed: 42]
  )

features = Explorer.DataFrame.discard(df, ["label", "group_label"])
preds = Hinoki.predict(booster, features)
```

## Cross-Validation

Cross-validation uses each fold's validation data for early stopping.
`Hinoki.CV.k_fold/2` returns fold best results plus aggregate stats.

```elixir
Hinoki.CV.k_fold({features, labels},
  k: 5,
  folding_rule: :stratified_shuffle,
  seed: 42,
  max_concurrency: 5,
  early_stopping_rounds: 10,
  num_iterations: 500,
  params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42]
)

Hinoki.CV.k_fold(df,
  target: :label,
  k: 5,
  early_stopping_rounds: 10,
  num_iterations: 500,
  params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42]
)

Hinoki.CV.grid_search(
  {features, labels},
  [learning_rate: [0.03, 0.1], num_leaves: [15, 31]],
  k: 5,
  max_concurrency: 5,
  early_stopping_rounds: 10,
  num_iterations: 500,
  params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42]
)
```

## Persistence

Saving to a path with an extension writes a raw LightGBM text model.
Saving to a directory, or to a new path without an extension, writes a
Hinoki bundle: `model.txt` stores the raw LightGBM model and
`hinoki.json` stores Hinoki metadata such as early stopping results.
Loading the directory restores both:

```elixir
Hinoki.save(booster, "path/to/model.txt")

Hinoki.save(booster, "path/to/model_dir")
loaded = Hinoki.load("path/to/model_dir")
Hinoki.best(loaded)
```

## Notes

- `Hinoki.CV.k_fold/2` requires `:early_stopping_rounds` and builds each
  fold's validation data internally.

## Implementation

Hinoki links against LightGBM 4.3.0 and calls its C API through a NIF.
Model resources are managed automatically.

## License

Apache License 2.0. See [LICENSE](LICENSE).
