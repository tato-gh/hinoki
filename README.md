# Hinoki ~ LightGBM bindings for Elixir.

Status: pre-alpha. The current surface covers `train`, `predict`,
`save`/`load`, `dump`/`load_string`, basic booster introspection, and
early stopping. Cross-validation is deferred.

## Architecture

- A single NIF talks to LightGBM's C-API. No CLI.
- All heavy operations are scheduled as dirty NIFs (`ERL_NIF_DIRTY_JOB_CPU_BOUND`
  for compute, `ERL_NIF_DIRTY_JOB_IO_BOUND` for serialization).
- Boosters and datasets are held as Erlang resource objects. The
  destructor calls `LGBM_BoosterFree` / `LGBM_DatasetFree`, so process
  death cleans up automatically — no manual `free`.
- The native build pulls a pinned LightGBM commit, builds it via CMake,
  and stages `lib_lightgbm.so` next to the NIF using `rpath=$ORIGIN/lib`.

## Threading

LightGBM uses OpenMP internally and runs *outside* the BEAM scheduler
on the dirty thread pool. By default it tries to use every CPU on the
machine, which can starve the BEAM. **Always set `num_threads`
explicitly** to a value compatible with your scheduler topology.

## Usage

```elixir
{features, labels} = some_data()

booster =
  Hinoki.train({features, labels},
    num_iterations: 100,
    params: [objective: "binary", num_threads: 4, learning_rate: 0.05, seed: 42]
  )

preds = Hinoki.predict(booster, features)
gain = Hinoki.feature_importance(booster)
split = Hinoki.feature_importance(booster, :split)

Hinoki.num_features(booster)
Hinoki.current_iteration(booster)
Hinoki.categorical_features(booster)
Hinoki.save(booster, "path/to/model.txt")
Hinoki.save(booster, "path/to/model_dir")
```

Early stopping uses validation data:

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

Saving to a directory writes a Hinoki bundle: `model.txt` stores the raw
LightGBM model and `hinoki.json` stores Hinoki metadata such as early stopping
results. Loading the directory restores both:

```elixir
Hinoki.save(booster, "path/to/model_dir")
loaded = Hinoki.load("path/to/model_dir")
Hinoki.best(loaded)
```

DataFrames work too:

```elixir
Hinoki.train(df, target: :label, params: [objective: "regression", num_threads: 1, seed: 42])
```

DataFrame columns with Explorer's `:category` dtype are passed to
LightGBM as categorical features automatically. Tensor input has no
column dtype metadata, so pass `categorical_feature` in `:params` when
using `Nx.Tensor` features. `Hinoki.categorical_features/1` returns
the 0-based feature indexes marked as categorical in the trained model.

## Reproducibility

Floating-point training results are only bit-identical with
`num_threads: 1` plus a fixed `seed`. Tests rely on this.
