defmodule Hinoki.TestData do
  @moduledoc false

  @doc """
  Tiny linearly-separable binary classification fixture.

  Returns `{features, labels}` as `Nx.Tensor`s. Class 0 lives near
  `(0, 0)`, class 1 near `(1, 1)`. Deterministic.
  """
  def binary_xor_like(n_per_class \\ 50) do
    key = Nx.Random.key(0)

    {a, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {n_per_class, 2})
    {b, _key} = Nx.Random.normal(key, 1.0, 0.1, shape: {n_per_class, 2})

    features = Nx.concatenate([a, b], axis: 0)

    labels =
      Nx.concatenate([
        Nx.broadcast(0.0, {n_per_class}),
        Nx.broadcast(1.0, {n_per_class})
      ])

    {features, labels}
  end

  @doc """
  Fixed-seed training options. Use these in tests where you want
  bit-identical results across runs.
  """
  def deterministic_params do
    [objective: "binary", num_threads: 1, seed: 42, verbose: -1]
  end
end
