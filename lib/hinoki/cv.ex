defmodule Hinoki.CV do
  @moduledoc """
  Cross-validation helpers for Hinoki models.
  """

  @type cv_result :: %{
          folds: [Hinoki.Booster.best()],
          stats: %{
            metric: binary(),
            score: %{mean: float(), std: float()},
            iteration: %{mean: float(), std: float()}
          }
        }

  @doc """
  Run k-fold cross-validation for tensor or DataFrame training data.

  Each fold trains with `Hinoki.train/2`, using the fold holdout as
  validation data. `:early_stopping_rounds` is required, and each fold result
  is the same map shape returned by `Hinoki.best/1`.

  ## Options

    * `:k` - number of folds. Defaults to `5`.
    * `:folding_rule` - one of `:raw`, `:shuffle`, `:stratified`, or
      `:stratified_shuffle`. Defaults to `:raw`.
    * `:seed` - integer seed for reproducible shuffled folding rules.
    * `:max_concurrency` - max number of folds to run concurrently. Defaults to `1`.
    * `:early_stopping_rounds` - required.
    * `:target` - required when input is an `Explorer.DataFrame`.
    * all other options are forwarded to `Hinoki.train/2`.
  """
  @spec k_fold({Nx.Tensor.t(), Nx.Tensor.t()} | Explorer.DataFrame.t(), keyword()) :: cv_result()
  def k_fold(input, opts \\ [])

  def k_fold({%Nx.Tensor{} = features, %Nx.Tensor{} = labels}, opts) when is_list(opts) do
    if Keyword.has_key?(opts, :valid) do
      raise ArgumentError, "Hinoki.CV.k_fold/2 builds validation folds; do not pass :valid"
    end

    unless Keyword.has_key?(opts, :early_stopping_rounds) do
      raise ArgumentError, "Hinoki.CV.k_fold/2 requires :early_stopping_rounds"
    end

    nrow = validate_tensor_input!(features, labels)
    k = validate_k!(Keyword.get(opts, :k, 5), nrow)
    folding_rule = validate_folding_rule!(Keyword.get(opts, :folding_rule, :raw))
    seed = validate_seed!(Keyword.get(opts, :seed))
    max_concurrency = validate_max_concurrency!(Keyword.get(opts, :max_concurrency, 1), k)
    train_opts = train_opts(opts)

    folds =
      fold_indices(nrow, k, fn -> tensor_to_list(labels) end, folding_rule, seed)
      |> run_folds(max_concurrency, fn {train_idx, valid_idx} ->
        train_input = {take_rows(features, train_idx), take_rows(labels, train_idx)}
        valid_input = {take_rows(features, valid_idx), take_rows(labels, valid_idx)}

        train_input
        |> Hinoki.train(Keyword.put(train_opts, :valid, valid_input))
        |> Hinoki.best()
      end)

    %{folds: folds, stats: stats(folds)}
  end

  def k_fold(%Explorer.DataFrame{} = df, opts) when is_list(opts) do
    if Keyword.has_key?(opts, :valid) do
      raise ArgumentError, "Hinoki.CV.k_fold/2 builds validation folds; do not pass :valid"
    end

    unless Keyword.has_key?(opts, :early_stopping_rounds) do
      raise ArgumentError, "Hinoki.CV.k_fold/2 requires :early_stopping_rounds"
    end

    unless Keyword.has_key?(opts, :target) do
      raise ArgumentError, "DataFrame k-fold cross-validation requires the :target option"
    end

    nrow = Explorer.DataFrame.n_rows(df)
    k = validate_k!(Keyword.get(opts, :k, 5), nrow)
    folding_rule = validate_folding_rule!(Keyword.get(opts, :folding_rule, :raw))
    seed = validate_seed!(Keyword.get(opts, :seed))
    max_concurrency = validate_max_concurrency!(Keyword.get(opts, :max_concurrency, 1), k)
    target = Keyword.fetch!(opts, :target)
    labels_fun = fn -> df |> Explorer.DataFrame.pull(target) |> Explorer.Series.to_list() end
    train_opts = train_opts(opts)

    folds =
      fold_indices(nrow, k, labels_fun, folding_rule, seed)
      |> run_folds(max_concurrency, fn {train_idx, valid_idx} ->
        train_df = Explorer.DataFrame.slice(df, train_idx)
        valid_df = Explorer.DataFrame.slice(df, valid_idx)

        train_df
        |> Hinoki.train(Keyword.put(train_opts, :valid, valid_df))
        |> Hinoki.best()
      end)

    %{folds: folds, stats: stats(folds)}
  end

  def k_fold(other, _opts) do
    raise ArgumentError,
          "expected an Explorer.DataFrame or {features, labels} tensor tuple for k-fold cross-validation, got: #{inspect(other)}"
  end

  @doc """
  Run k-fold cross-validation for every parameter combination in `grid`.

  `grid` must be a keyword list or map whose values are non-empty lists. Fixed
  parameters are read from `opts[:params]`; grid parameters override fixed
  parameters with the same key.

  This function returns all results and does not choose the best result.
  """
  @spec grid_search(
          {Nx.Tensor.t(), Nx.Tensor.t()} | Explorer.DataFrame.t(),
          keyword() | map(),
          keyword()
        ) :: %{
          results: [%{params: keyword(), cv: cv_result()}]
        }
  def grid_search(input, grid, opts \\ []) when is_list(opts) do
    base_params = Keyword.get(opts, :params, [])

    unless is_list(base_params) do
      raise ArgumentError, "expected :params to be a keyword list, got: #{inspect(base_params)}"
    end

    results =
      grid
      |> grid_combinations()
      |> Enum.map(fn grid_params ->
        params = Keyword.merge(base_params, grid_params)
        cv = k_fold(input, Keyword.put(opts, :params, params))

        %{params: params, cv: cv}
      end)

    %{results: results}
  end

  defp validate_tensor_input!(features, labels) do
    case {Nx.shape(features), Nx.shape(labels)} do
      {{nrow, _ncol}, {nrow}} ->
        nrow

      {feature_shape, label_shape} ->
        raise ArgumentError,
              "expected features shape {nrow, ncol} and labels shape {nrow}, got: #{inspect(feature_shape)} and #{inspect(label_shape)}"
    end
  end

  defp validate_k!(k, nrow) when is_integer(k) and k >= 2 and k <= nrow, do: k

  defp validate_k!(k, nrow) do
    raise ArgumentError, "expected :k to be an integer between 2 and #{nrow}, got: #{inspect(k)}"
  end

  defp validate_max_concurrency!(max_concurrency, _k)
       when is_integer(max_concurrency) and max_concurrency >= 1 do
    max_concurrency
  end

  defp validate_max_concurrency!(max_concurrency, _k) do
    raise ArgumentError,
          "expected :max_concurrency to be a positive integer, got: #{inspect(max_concurrency)}"
  end

  defp validate_folding_rule!(rule)
       when rule in [:raw, :shuffle, :stratified, :stratified_shuffle],
       do: rule

  defp validate_folding_rule!(rule) do
    raise ArgumentError,
          "expected :folding_rule to be one of :raw, :shuffle, :stratified, or :stratified_shuffle, got: #{inspect(rule)}"
  end

  defp validate_seed!(nil), do: nil
  defp validate_seed!(seed) when is_integer(seed), do: seed

  defp validate_seed!(seed) do
    raise ArgumentError, "expected :seed to be an integer, got: #{inspect(seed)}"
  end

  defp train_opts(opts) do
    opts
    |> Keyword.delete(:k)
    |> Keyword.delete(:folding_rule)
    |> Keyword.delete(:seed)
    |> Keyword.delete(:max_concurrency)
  end

  defp run_folds(folds, 1, fun), do: Enum.map(folds, fun)

  defp run_folds(folds, max_concurrency, fun) do
    folds
    |> Task.async_stream(fun,
      max_concurrency: max_concurrency,
      ordered: true,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, fold} -> fold end)
  end

  defp fold_indices(nrow, k, _labels_fun, :raw, _seed) do
    indices = Enum.to_list(0..(nrow - 1))
    split_indices(indices, k)
  end

  defp fold_indices(nrow, k, _labels_fun, :shuffle, seed) do
    indices =
      0..(nrow - 1)
      |> Enum.to_list()
      |> shuffle_list(seed)

    split_indices(indices, k)
  end

  defp fold_indices(nrow, k, labels_fun, :stratified, _seed) do
    stratified_indices(nrow, labels_fun.(), k, false, nil)
  end

  defp fold_indices(nrow, k, labels_fun, :stratified_shuffle, seed) do
    stratified_indices(nrow, labels_fun.(), k, true, seed)
  end

  defp split_indices(indices, k) do
    length(indices)
    |> fold_ranges(k)
    |> Enum.map(fn {offset, size} ->
      valid_idx = Enum.slice(indices, offset, size)
      valid_set = MapSet.new(valid_idx)
      train_idx = Enum.reject(indices, &MapSet.member?(valid_set, &1))

      {train_idx, valid_idx}
    end)
  end

  defp stratified_indices(nrow, labels, k, shuffle?, seed) do
    unless length(labels) == nrow do
      raise ArgumentError,
            "expected labels length to match row count, got #{length(labels)} labels for #{nrow} rows"
    end

    labels
    |> Enum.with_index()
    |> Enum.group_by(fn {label, _idx} -> label end, fn {_label, idx} -> idx end)
    |> Enum.sort_by(fn {label, _indices} -> label end)
    |> Enum.map(fn {_label, indices} -> indices end)
    |> validate_stratified_groups!(k)
    |> Enum.with_index()
    |> Enum.map(fn indices ->
      {indices, group_index} = indices
      indices = if shuffle?, do: shuffle_list(indices, seed, group_index), else: indices
      validation_parts(indices, k)
    end)
    |> Enum.reduce(List.duplicate([], k), fn parts, folds ->
      Enum.zip_with(folds, parts, &(&1 ++ &2))
    end)
    |> Enum.map(fn valid_idx ->
      valid_set = MapSet.new(valid_idx)

      train_idx =
        0..(nrow - 1)
        |> Enum.reject(&MapSet.member?(valid_set, &1))

      {train_idx, valid_idx}
    end)
  end

  defp validate_stratified_groups!(groups, k) do
    smallest_undersized_group_size =
      groups
      |> Enum.map(&length/1)
      |> Enum.filter(&(&1 < k))
      |> min_or_nil()

    if smallest_undersized_group_size do
      raise ArgumentError,
            "stratified k-fold requires every label group to have at least k rows; smallest group has #{smallest_undersized_group_size} rows for k=#{k}"
    end

    groups
  end

  defp validation_parts(indices, k) do
    indices
    |> split_indices(k)
    |> Enum.map(fn {_train_idx, valid_idx} -> valid_idx end)
  end

  defp fold_ranges(nrow, k) do
    base_size = div(nrow, k)
    extra = rem(nrow, k)

    {folds, _offset} =
      Enum.map_reduce(0..(k - 1), 0, fn fold, offset ->
        size = base_size + if(fold < extra, do: 1, else: 0)
        {{offset, size}, offset + size}
      end)

    folds
  end

  defp take_rows(tensor, indices) do
    Nx.take(tensor, Nx.tensor(indices, type: :s64), axis: 0)
  end

  defp tensor_to_list(tensor) do
    tensor
    |> Nx.to_flat_list()
  end

  defp min_or_nil([]), do: nil
  defp min_or_nil(values), do: Enum.min(values)

  defp shuffle_list(values, seed, salt \\ 0)

  defp shuffle_list(values, nil, salt) do
    seed =
      :erlang.unique_integer([:positive])
      |> Kernel.+(:erlang.phash2({self(), System.monotonic_time(), salt}))

    shuffle_list(values, seed, salt)
  end

  defp shuffle_list(values, seed, salt) do
    state = :rand.seed_s(:exsss, {seed, seed + salt + 1, seed + salt + 2})

    {tagged, _state} =
      Enum.map_reduce(values, state, fn value, state ->
        {key, state} = :rand.uniform_s(state)
        {{key, value}, state}
      end)

    tagged
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(&elem(&1, 1))
  end

  defp grid_combinations(grid) when is_map(grid) do
    grid
    |> Map.to_list()
    |> grid_combinations()
  end

  defp grid_combinations(grid) when is_list(grid) do
    unless Keyword.keyword?(grid) do
      raise ArgumentError, "expected grid to be a keyword list or map, got: #{inspect(grid)}"
    end

    Enum.reduce(grid, [[]], fn {key, values}, combinations ->
      unless is_list(values) and values != [] do
        raise ArgumentError,
              "expected grid value for #{inspect(key)} to be a non-empty list, got: #{inspect(values)}"
      end

      for combination <- combinations, value <- values do
        Keyword.put(combination, key, value)
      end
    end)
  end

  defp grid_combinations(grid) do
    raise ArgumentError, "expected grid to be a keyword list or map, got: #{inspect(grid)}"
  end

  defp stats([%{metric: metric} | _] = folds) do
    unless Enum.all?(folds, &(&1.metric == metric)) do
      metrics = folds |> Enum.map(& &1.metric) |> Enum.uniq()
      raise ArgumentError, "expected all folds to use the same metric, got: #{inspect(metrics)}"
    end

    %{
      metric: metric,
      score: values_stats(Enum.map(folds, & &1.score)),
      iteration: values_stats(Enum.map(folds, &(&1.iteration * 1.0)))
    }
  end

  defp values_stats(values) do
    mean = Enum.sum(values) / length(values)

    variance =
      values
      |> Enum.map(&:math.pow(&1 - mean, 2))
      |> Enum.sum()
      |> Kernel./(length(values))

    %{mean: mean, std: :math.sqrt(variance)}
  end
end
