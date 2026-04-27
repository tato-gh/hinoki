# Hinoki ~ LightGBM bindings for Elixir.

Status: pre-alpha. The current surface covers `train`, `predict`,
`save`/`load`, `dump`/`load_string`, and basic booster introspection.
Cross-validation and early stopping are deferred.

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
Hinoki.save(booster, "model.txt")
```

DataFrames work too:

```elixir
Hinoki.train(df, target: :label, params: [objective: "regression", num_threads: 1, seed: 42])
```

## Reproducibility

Floating-point training results are only bit-identical with
`num_threads: 1` plus a fixed `seed`. Tests rely on this.
