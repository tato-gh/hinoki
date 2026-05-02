defmodule Hinoki do
  @bundle_model_filename "model.txt"
  @bundle_metadata_filename "hinoki.json"

  @moduledoc """
  LightGBM bindings for Elixir.

  ## Quick start

      df = Explorer.Datasets.iris() |> Explorer.DataFrame.discard(["species"])

      booster =
        Hinoki.train(df,
          target: :sepal_length,
          num_iterations: 50,
          params: [objective: "regression", num_threads: 1, seed: 42]
        )

      preds =
        df
        |> Explorer.DataFrame.discard(["sepal_length"])
        |> then(&Hinoki.predict(booster, &1))

      Nx.shape(preds)
      #=> {150}

  ## Inputs

  `train/2` accepts either an `Explorer.DataFrame` (with a `:target`
  option naming the label column) or a `{features, labels}` tuple of
  `Nx.Tensor`s. `predict/2` accepts a `DataFrame` (using all columns)
  or an `Nx.Tensor`.

  ## Parameters

  Parameters are forwarded verbatim to LightGBM. See
  https://lightgbm.readthedocs.io/en/latest/Parameters.html for the
  full list. When deterministic results are needed, set both
  `num_threads: 1` and a `seed`. For `Explorer.DataFrame` training,
  `:category` feature columns are forwarded to LightGBM automatically
  unless `categorical_feature` is already present in `:params`.
  """

  alias Hinoki.{Booster, NIF}

  @default_num_iterations 100

  @doc """
  Train a LightGBM booster.

  ## Inputs

    * `Explorer.DataFrame.t()` — pass `target: column_name` to nominate the label column.
    * `{Nx.Tensor.t(), Nx.Tensor.t()}` — `{features, labels}` where features
      has shape `{nrow, ncol}` and labels has shape `{nrow}`.

  ## Options

    * `:num_iterations` — boosting rounds. Default `#{@default_num_iterations}`.
    * `:params` — keyword/map of LightGBM parameters (e.g. `objective`, `learning_rate`).
    * `:target` — required when input is a `DataFrame`.
    * `:group` — ranking group sizes for tensor input, or a group column name
      for DataFrame input. DataFrame group columns are excluded from features
      and converted to contiguous group sizes.
    * `:valid` — validation data for early stopping. Pass a `{features, labels}`
      tuple or a DataFrame. DataFrame validation uses the same `:target` column
      as training.
    * `:valid_group` — validation ranking group sizes for tensor input, or a
      validation group column name for DataFrame input. DataFrame validation
      defaults to the `:group` column when `:valid_group` is not set.
    * `:early_stopping_rounds` — stop training when the first metric on the
      validation dataset does not improve for this many rounds.
  """
  @spec train(term(), keyword()) :: Booster.t()
  def train(input, opts \\ []) do
    num_iter = Keyword.get(opts, :num_iterations, @default_num_iterations)
    params = Keyword.get(opts, :params, [])
    target = Keyword.get(opts, :target)
    group = Keyword.get(opts, :group)
    valid_group = Keyword.get(opts, :valid_group, default_valid_group(input, group))
    early_stopping_rounds = Keyword.get(opts, :early_stopping_rounds)

    {features_bin, labels_bin, group_bin, nrow, ncol, categorical_indices} =
      to_train_payload(input, target, group)

    valid_payload = to_valid_payload(Keyword.get(opts, :valid), input, target, valid_group, ncol)
    params = maybe_put_categorical_feature(params, categorical_indices)
    params_bin = encode_params(params)

    dataset_ref = create_dataset!(features_bin, labels_bin, group_bin, nrow, ncol, params_bin)

    booster_ref = unwrap!(NIF.booster_create(dataset_ref, params_bin))

    valid_ref =
      case valid_payload do
        nil ->
          nil

        {valid_features_bin, valid_labels_bin, valid_group_bin, valid_nrow, valid_ncol} ->
          ref =
            create_dataset!(
              valid_features_bin,
              valid_labels_bin,
              valid_group_bin,
              valid_nrow,
              valid_ncol,
              params_bin,
              dataset_ref
            )

          unwrap!(NIF.booster_add_valid_data(booster_ref, ref))
          ref
      end

    best = run_training!(booster_ref, num_iter, early_stopping_rounds, valid_payload, valid_ref)

    %Booster{ref: booster_ref, best: best}
  end

  @doc """
  Run inference with a trained booster.

  Returns a `Nx.Tensor`. For binary classification and regression the
  shape is `{nrow}`; for multiclass classification it is `{nrow, num_classes}`.
  """
  @spec predict(Booster.t(), term(), keyword()) :: Nx.Tensor.t()
  def predict(%Booster{ref: ref}, input, opts \\ []) do
    params = Keyword.get(opts, :params, [])
    {features_bin, nrow, ncol} = to_predict_payload(input)
    params_bin = encode_params(params)

    out_bin =
      unwrap!(NIF.booster_predict_for_mat(ref, features_bin, nrow, ncol, params_bin))

    decode_predictions(out_bin, nrow)
  end

  @doc """
  Persist a booster.

  File paths are written in LightGBM's raw text model format. Directory paths
  are written as a Hinoki bundle with `model.txt` and `hinoki.json`, preserving
  Hinoki metadata such as `best`. When saving to a new path,
  paths without an extension are treated as bundle directories.
  """
  @spec save(Booster.t(), Path.t()) :: :ok
  def save(%Booster{} = booster, path) do
    if bundle_path?(path) do
      save_bundle(booster, path)
    else
      File.write!(path, dump(booster))
    end
  end

  @doc "Load a booster previously written by `save/2`."
  @spec load(Path.t()) :: Booster.t()
  def load(path) do
    if File.dir?(path) do
      load_bundle(path)
    else
      path |> File.read!() |> load_string()
    end
  end

  @doc "Serialize a booster to LightGBM's text model format as a binary."
  @spec dump(Booster.t()) :: binary()
  def dump(%Booster{ref: ref}) do
    unwrap!(NIF.booster_save_model_to_string(ref, 0, -1, 0))
  end

  @doc "Restore a booster from a binary previously produced by `dump/1`."
  @spec load_string(binary()) :: Booster.t()
  def load_string(model_bin) when is_binary(model_bin) do
    ref = unwrap!(NIF.booster_load_model_from_string(model_bin))
    %Booster{ref: ref}
  end

  @doc "Return the linked LightGBM library version, if available."
  @spec version() :: binary()
  def version, do: NIF.lgbm_version()

  @doc """
  Return booster metadata or derived information.

  Supported keys:

    * `:num_features`
    * `:num_classes`
    * `:current_iteration`
    * `:params`
    * `:best`
    * `:categorical_features`
    * `:feature_importance` — equivalent to `{:feature_importance, :gain}`
    * `{:feature_importance, :gain}`
    * `{:feature_importance, :split}`
  """
  @spec info(Booster.t(), atom() | tuple()) ::
          integer()
          | float()
          | nil
          | map()
          | [non_neg_integer()]
          | Booster.best()
          | Nx.Tensor.t()
  def info(%Booster{ref: ref}, :num_features) do
    unwrap!(NIF.booster_get_num_feature(ref))
  end

  def info(%Booster{ref: ref}, :num_classes) do
    unwrap!(NIF.booster_get_num_classes(ref))
  end

  def info(%Booster{ref: ref}, :current_iteration) do
    unwrap!(NIF.booster_get_current_iteration(ref))
  end

  def info(%Booster{} = booster, :params) do
    booster
    |> dump()
    |> parse_model_params()
  end

  def info(%Booster{} = booster, :best) do
    best(booster)
  end

  def info(%Booster{} = booster, :categorical_features) do
    categorical_features(booster)
  end

  def info(%Booster{} = booster, :feature_importance) do
    feature_importance(booster)
  end

  def info(%Booster{} = booster, {:feature_importance, type}) do
    feature_importance(booster, type)
  end

  def info(%Booster{}, key) do
    raise ArgumentError, "unsupported booster info key: #{inspect(key)}"
  end

  @doc "Return the number of features the booster was trained with."
  @spec num_features(Booster.t()) :: non_neg_integer()
  def num_features(%Booster{} = booster), do: info(booster, :num_features)

  @doc "Return the number of classes known to the booster."
  @spec num_classes(Booster.t()) :: pos_integer()
  def num_classes(%Booster{} = booster), do: info(booster, :num_classes)

  @doc "Return the current boosting iteration."
  @spec current_iteration(Booster.t()) :: non_neg_integer()
  def current_iteration(%Booster{} = booster), do: info(booster, :current_iteration)

  @doc "Return the best validation result observed during early stopping, or nil."
  @spec best(Booster.t()) :: Booster.best() | nil
  def best(%Booster{best: best}), do: best

  @doc "Return 0-based feature indexes marked as categorical in the booster."
  @spec categorical_features(Booster.t()) :: [non_neg_integer()]
  def categorical_features(%Booster{} = booster) do
    booster
    |> dump()
    |> parse_categorical_features()
  end

  @doc """
  Return feature importance as an `Nx.Tensor`.

  `type` may be `:gain` or `:split`. Gain importance is returned as `f64`;
  split importance is returned as `s64`.
  """
  @spec feature_importance(Booster.t(), :gain | :split) :: Nx.Tensor.t()
  def feature_importance(%Booster{ref: ref}, type \\ :gain) do
    {importance_type, nx_type} = importance_type!(type)

    ref
    |> NIF.booster_feature_importance(0, importance_type)
    |> unwrap!()
    |> Nx.from_binary(:f64)
    |> maybe_cast_importance(nx_type)
  end

  @doc """
  Return feature importance paired with caller-provided feature names.

  `feature_names` must contain exactly one name per feature in the booster.
  `type` accepts the same values as `feature_importance/2`.
  """
  @spec named_feature_importance(Booster.t(), Enumerable.t(), :gain | :split) :: [
          {term(), number()}
        ]
  def named_feature_importance(%Booster{} = booster, feature_names, type \\ :gain) do
    feature_names = Enum.to_list(feature_names)
    expected = num_features(booster)
    actual = length(feature_names)

    unless actual == expected do
      raise ArgumentError,
            "expected #{expected} feature names, got: #{actual}"
    end

    booster
    |> feature_importance(type)
    |> Nx.to_flat_list()
    |> then(&Enum.zip(feature_names, &1))
  end

  @doc """
  Measure validation score changes after shuffling feature columns.

  `metric_fn` is called as `metric_fn.(y, predictions)`. It may return a number
  or an `Nx` scalar. Scores are returned as-is; `delta` is `mean - baseline_score`,
  so callers can interpret the direction according to their metric.

  Options:

    * `:features` — subset of features to permute. Use indexes for tensor input
      and column names for DataFrame input.
    * `:n_repeats` — number of shuffles per feature. Default `5`.
    * `:seed` — integer seed for reproducible shuffles.
  """
  @spec permutation_importance(
          Booster.t(),
          Nx.Tensor.t() | Explorer.DataFrame.t(),
          term(),
          fun(),
          keyword()
        ) ::
          %{
            baseline_score: number(),
            results: [
              {term(), %{delta: number(), mean: number(), std: number(), scores: [number()]}}
            ]
          }
  def permutation_importance(%Booster{} = booster, x, y, metric_fn, opts \\ [])
      when is_function(metric_fn, 2) and is_list(opts) do
    n_repeats = validate_n_repeats!(Keyword.get(opts, :n_repeats, 5))
    {features, feature_names} = to_permutation_features(x)
    validate_feature_count!(elem(Nx.shape(features), 1), num_features(booster), "permutation")

    target_features =
      opts
      |> Keyword.get(:features, feature_names)
      |> normalize_permutation_target_features!(feature_names)

    baseline_score =
      booster
      |> predict(x)
      |> then(&metric_fn.(y, &1))
      |> metric_score_to_number!()

    seed = Keyword.get(opts, :seed)

    results =
      Enum.map(target_features, fn {feature, col_idx} ->
        scores =
          for repeat <- 1..n_repeats do
            shuffled_features = shuffle_tensor_column(features, col_idx, seed, repeat)

            booster
            |> predict(shuffled_features)
            |> then(&metric_fn.(y, &1))
            |> metric_score_to_number!()
          end

        mean = mean(scores)

        {feature,
         %{
           delta: mean - baseline_score,
           mean: mean,
           std: std(scores, mean),
           scores: scores
         }}
      end)

    %{baseline_score: baseline_score, results: results}
  end

  # ---------- input → binary ----------

  defp create_dataset!(
         features_bin,
         labels_bin,
         group_bin,
         nrow,
         ncol,
         params_bin,
         reference_ref \\ nil
       ) do
    dataset_ref =
      if is_nil(reference_ref) do
        unwrap!(NIF.dataset_create_from_mat(features_bin, nrow, ncol, params_bin))
      else
        unwrap!(
          NIF.dataset_create_from_mat_reference(
            features_bin,
            nrow,
            ncol,
            params_bin,
            reference_ref
          )
        )
      end

    unwrap!(NIF.dataset_set_label(dataset_ref, labels_bin))
    if group_bin, do: unwrap!(NIF.dataset_set_group(dataset_ref, group_bin))
    dataset_ref
  end

  defp run_training!(booster_ref, num_iter, nil, _valid_payload, _valid_ref) do
    unwrap!(NIF.booster_update_iters(booster_ref, num_iter))
    nil
  end

  defp run_training!(booster_ref, num_iter, early_stopping_rounds, valid_payload, _valid_ref) do
    early_stopping_rounds = validate_early_stopping_rounds!(early_stopping_rounds)

    if is_nil(valid_payload) do
      raise ArgumentError, ":early_stopping_rounds requires a validation dataset in :valid"
    end

    {best_iteration, best_score, metric_name, history} =
      unwrap!(
        NIF.booster_update_iters_early_stopping(booster_ref, num_iter, early_stopping_rounds)
      )

    %{
      iteration: best_iteration,
      score: best_score,
      metric: metric_name,
      history: history
    }
  end

  defp bundle_path?(path) do
    File.dir?(path) or
      (not File.exists?(path) and Path.extname(path) == "")
  end

  defp save_bundle(%Booster{} = booster, path) do
    File.mkdir_p!(path)
    File.write!(Path.join(path, @bundle_model_filename), dump(booster))

    metadata = %{"best" => encode_best(booster.best)}

    File.write!(Path.join(path, @bundle_metadata_filename), :json.encode(metadata))
  end

  defp load_bundle(path) do
    booster = Path.join(path, @bundle_model_filename) |> File.read!() |> load_string()

    metadata =
      path
      |> Path.join(@bundle_metadata_filename)
      |> File.read!()
      |> :json.decode()

    %Booster{booster | best: decode_best(Map.get(metadata, "best"))}
  end

  defp encode_best(nil), do: nil

  defp encode_best(%{iteration: iteration, score: score, metric: metric, history: history}) do
    %{
      "iteration" => iteration,
      "score" => score,
      "metric" => metric,
      "history" => history
    }
  end

  defp decode_best(nil), do: nil

  defp decode_best(%{
         "iteration" => iteration,
         "score" => score,
         "metric" => metric,
         "history" => history
       }) do
    %{
      iteration: iteration,
      score: score,
      metric: metric,
      history: history
    }
  end

  defp default_valid_group(%Explorer.DataFrame{}, group), do: group
  defp default_valid_group(_input, nil), do: nil
  defp default_valid_group(_input, _group), do: :__hinoki_missing_valid_group__

  defp to_train_payload({%Nx.Tensor{} = features, %Nx.Tensor{} = labels}, _target, group) do
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    labels_bin = labels_tensor_to_bin(labels, nrow)
    group_bin = group_to_bin(group, nrow, ":group")
    {features_bin, labels_bin, group_bin, nrow, ncol, []}
  end

  defp to_train_payload(%Explorer.DataFrame{} = df, target, group)
       when (is_atom(target) and not is_nil(target)) or is_binary(target) do
    target = to_string(target)
    group_col = normalize_optional_column(group, ":group")
    names = Explorer.DataFrame.names(df)

    unless target in names do
      raise ArgumentError,
            "target column #{inspect(target)} not found in DataFrame; available columns: #{inspect(names)}"
    end

    validate_group_column!(group_col, names)
    feature_cols = names -- Enum.reject([target, group_col], &is_nil/1)
    dtypes = Explorer.DataFrame.dtypes(df)

    if feature_cols == [] do
      raise ArgumentError,
            "DataFrame has no feature columns after dropping target/group columns"
    end

    features = df_columns_to_tensor(df, feature_cols)
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    categorical_indices = categorical_feature_indices(feature_cols, dtypes)
    group_bin = dataframe_group_to_bin(df, group_col, nrow)

    labels =
      df
      |> Explorer.DataFrame.pull(target)
      |> Explorer.Series.cast({:f, 32})
      |> Explorer.Series.to_tensor()

    labels_bin = labels_tensor_to_bin(labels, nrow)
    {features_bin, labels_bin, group_bin, nrow, ncol, categorical_indices}
  end

  defp to_train_payload(%Explorer.DataFrame{}, nil, _group) do
    raise ArgumentError,
          "training from a DataFrame requires the :target option naming the label column"
  end

  defp to_train_payload(other, _target, _group) do
    raise ArgumentError,
          "expected an Explorer.DataFrame or {features, labels} tensor tuple, got: #{inspect(other)}"
  end

  defp to_valid_payload(nil, _train_input, _target, _valid_group, _expected_ncol), do: nil

  defp to_valid_payload(
         {%Nx.Tensor{}, %Nx.Tensor{}},
         _train_input,
         _target,
         :__hinoki_missing_valid_group__,
         _expected_ncol
       ) do
    raise ArgumentError,
          ":valid_group is required when tensor validation data is used with training :group"
  end

  defp to_valid_payload(
         {%Nx.Tensor{} = features, %Nx.Tensor{} = labels},
         _train_input,
         _target,
         valid_group,
         expected_ncol
       ) do
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    validate_feature_count!(ncol, expected_ncol, "validation")
    labels_bin = labels_tensor_to_bin(labels, nrow)
    group_bin = group_to_bin(valid_group, nrow, ":valid_group")
    {features_bin, labels_bin, group_bin, nrow, ncol}
  end

  defp to_valid_payload(
         %Explorer.DataFrame{} = df,
         %Explorer.DataFrame{},
         target,
         valid_group,
         expected_ncol
       ) do
    {features_bin, labels_bin, group_bin, nrow, ncol, _categorical_indices} =
      to_train_payload(df, target, valid_group)

    validate_feature_count!(ncol, expected_ncol, "validation")
    {features_bin, labels_bin, group_bin, nrow, ncol}
  end

  defp to_valid_payload(
         %Explorer.DataFrame{},
         _train_input,
         _target,
         _valid_group,
         _expected_ncol
       ) do
    raise ArgumentError,
          "DataFrame validation data is only supported when training input is a DataFrame"
  end

  defp to_valid_payload(other, _train_input, _target, _valid_group, _expected_ncol) do
    raise ArgumentError,
          "expected :valid to be an Explorer.DataFrame or {features, labels} tensor tuple, got: #{inspect(other)}"
  end

  defp to_predict_payload(%Nx.Tensor{} = features) do
    tensor_to_features_bin(features)
  end

  defp to_predict_payload(%Explorer.DataFrame{} = df) do
    cols = Explorer.DataFrame.names(df)
    features = df_columns_to_tensor(df, cols)
    tensor_to_features_bin(features)
  end

  defp to_predict_payload(other) do
    raise ArgumentError,
          "expected an Explorer.DataFrame or Nx.Tensor for prediction, got: #{inspect(other)}"
  end

  defp df_columns_to_tensor(df, cols) do
    cols
    |> Enum.map(fn col ->
      series = Explorer.DataFrame.pull(df, col)

      if Explorer.Series.dtype(series) == :category do
        series
        |> Explorer.Series.to_tensor()
        |> Nx.as_type(:f64)
      else
        series
        |> Explorer.Series.cast({:f, 64})
        |> Explorer.Series.to_tensor()
      end
    end)
    |> Nx.stack(axis: 1)
  end

  defp categorical_feature_indices(feature_cols, dtypes) do
    feature_cols
    |> Enum.with_index()
    |> Enum.flat_map(fn {col, idx} ->
      if Map.fetch!(dtypes, col) == :category, do: [idx], else: []
    end)
  end

  defp tensor_to_features_bin(%Nx.Tensor{} = t) do
    case Nx.shape(t) do
      {nrow, ncol} ->
        bin = t |> Nx.as_type(:f64) |> Nx.to_binary()
        {bin, nrow, ncol}

      other ->
        raise ArgumentError,
              "expected 2D feature tensor of shape {nrow, ncol}, got shape #{inspect(other)}"
    end
  end

  defp labels_tensor_to_bin(%Nx.Tensor{} = t, expected_nrow) do
    case Nx.shape(t) do
      {^expected_nrow} ->
        t |> Nx.as_type(:f32) |> Nx.to_binary()

      other ->
        raise ArgumentError,
              "labels tensor shape #{inspect(other)} does not match feature row count {#{expected_nrow}}"
    end
  end

  defp group_to_bin(nil, _expected_nrow, _option), do: nil

  defp group_to_bin(group, expected_nrow, option) when is_list(group) do
    validate_group!(group, expected_nrow, option)

    for value <- group, into: <<>> do
      <<value::signed-native-32>>
    end
  end

  defp group_to_bin(group, _expected_nrow, option) do
    raise ArgumentError,
          "expected #{option} to be a list of positive integer group sizes, got: #{inspect(group)}"
  end

  defp validate_group!(group, expected_nrow, option) do
    if group == [] do
      raise ArgumentError, "expected #{option} to contain at least one group size"
    end

    unless Enum.all?(group, &(is_integer(&1) and &1 > 0)) do
      raise ArgumentError,
            "expected #{option} to contain only positive integer group sizes, got: #{inspect(group)}"
    end

    actual_nrow = Enum.sum(group)

    unless actual_nrow == expected_nrow do
      raise ArgumentError,
            "#{option} row count #{actual_nrow} does not match feature row count #{expected_nrow}"
    end

    :ok
  end

  defp normalize_optional_column(nil, _option), do: nil
  defp normalize_optional_column(column, _option) when is_atom(column), do: to_string(column)
  defp normalize_optional_column(column, _option) when is_binary(column), do: column

  defp normalize_optional_column(column, option) do
    raise ArgumentError,
          "expected #{option} to be a DataFrame column name, got: #{inspect(column)}"
  end

  defp validate_group_column!(nil, _names), do: :ok

  defp validate_group_column!(group_col, names) do
    unless group_col in names do
      raise ArgumentError,
            "group column #{inspect(group_col)} not found in DataFrame; available columns: #{inspect(names)}"
    end
  end

  defp dataframe_group_to_bin(_df, nil, _nrow), do: nil

  defp dataframe_group_to_bin(df, group_col, nrow) do
    group =
      df
      |> Explorer.DataFrame.pull(group_col)
      |> Explorer.Series.to_list()
      |> contiguous_group_sizes!(group_col)

    group_to_bin(group, nrow, ":group")
  end

  defp contiguous_group_sizes!([], group_col) do
    raise ArgumentError, "group column #{inspect(group_col)} has no rows"
  end

  defp contiguous_group_sizes!([first | rest], group_col) do
    {sizes, _current_value, current_size, _seen} =
      Enum.reduce(rest, {[], first, 1, MapSet.new([first])}, fn value,
                                                                {sizes, current_value,
                                                                 current_size, seen} ->
        if value == current_value do
          {sizes, current_value, current_size + 1, seen}
        else
          if MapSet.member?(seen, value) do
            raise ArgumentError,
                  "group column #{inspect(group_col)} must be ordered by contiguous groups; value #{inspect(value)} appears in multiple blocks"
          end

          {[current_size | sizes], value, 1, MapSet.put(seen, value)}
        end
      end)

    Enum.reverse([current_size | sizes])
  end

  defp validate_feature_count!(ncol, ncol, _context), do: :ok

  defp validate_feature_count!(ncol, expected_ncol, context) do
    raise ArgumentError,
          "#{context} feature count #{ncol} does not match training feature count #{expected_ncol}"
  end

  defp to_permutation_features(%Nx.Tensor{} = features) do
    {_features_bin, _nrow, ncol} = tensor_to_features_bin(features)
    {features, Enum.to_list(0..(ncol - 1))}
  end

  defp to_permutation_features(%Explorer.DataFrame{} = df) do
    feature_names = Explorer.DataFrame.names(df)

    if feature_names == [] do
      raise ArgumentError, "DataFrame has no feature columns for permutation importance"
    end

    {df_columns_to_tensor(df, feature_names), feature_names}
  end

  defp to_permutation_features(other) do
    raise ArgumentError,
          "expected an Explorer.DataFrame or Nx.Tensor for permutation importance, got: #{inspect(other)}"
  end

  defp normalize_permutation_target_features!(features, feature_names) do
    features
    |> List.wrap()
    |> Enum.map(fn feature ->
      normalized_feature = normalize_permutation_feature(feature, feature_names)

      case Enum.find_index(feature_names, &(&1 == normalized_feature)) do
        nil ->
          raise ArgumentError,
                "permutation feature #{inspect(feature)} not found; available features: #{inspect(feature_names)}"

        col_idx ->
          {normalized_feature, col_idx}
      end
    end)
  end

  defp normalize_permutation_feature(feature, feature_names) do
    cond do
      feature in feature_names -> feature
      is_atom(feature) and to_string(feature) in feature_names -> to_string(feature)
      true -> feature
    end
  end

  defp validate_n_repeats!(n_repeats) when is_integer(n_repeats) and n_repeats > 0,
    do: n_repeats

  defp validate_n_repeats!(n_repeats) do
    raise ArgumentError,
          "expected :n_repeats to be a positive integer, got: #{inspect(n_repeats)}"
  end

  defp shuffle_tensor_column(features, col_idx, seed, repeat) do
    rows = Nx.to_list(features)

    shuffled_column =
      rows
      |> Enum.map(&Enum.at(&1, col_idx))
      |> shuffle_values(seed, col_idx, repeat)

    rows
    |> Enum.zip(shuffled_column)
    |> Enum.map(fn {row, value} -> List.replace_at(row, col_idx, value) end)
    |> Nx.tensor(type: Nx.type(features))
  end

  defp shuffle_values(values, nil, _col_idx, _repeat), do: Enum.shuffle(values)

  defp shuffle_values(values, seed, col_idx, repeat) when is_integer(seed) do
    rand_state = :rand.seed_s(:exsss, {seed, col_idx + 1, repeat})

    values
    |> Enum.map_reduce(rand_state, fn value, state ->
      {random, state} = :rand.uniform_s(state)
      {{random, value}, state}
    end)
    |> elem(0)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(&elem(&1, 1))
  end

  defp shuffle_values(_values, seed, _col_idx, _repeat) do
    raise ArgumentError, "expected :seed to be an integer, got: #{inspect(seed)}"
  end

  defp metric_score_to_number!(score) when is_number(score), do: score

  defp metric_score_to_number!(%Nx.Tensor{} = score) do
    case Nx.shape(score) do
      {} ->
        score |> Nx.to_number()

      shape ->
        raise ArgumentError,
              "expected metric_fn to return a number or Nx scalar, got tensor shape #{inspect(shape)}"
    end
  end

  defp metric_score_to_number!(score) do
    raise ArgumentError,
          "expected metric_fn to return a number or Nx scalar, got: #{inspect(score)}"
  end

  defp mean(values), do: Enum.sum(values) / length(values)

  defp std(values, mean) do
    variance =
      values
      |> Enum.map(fn value -> :math.pow(value - mean, 2) end)
      |> mean()

    :math.sqrt(variance)
  end

  defp decode_predictions(bin, nrow) do
    total_floats = div(byte_size(bin), 8)
    num_classes = div(total_floats, nrow)

    tensor = Nx.from_binary(bin, :f64)

    if num_classes == 1 do
      Nx.reshape(tensor, {nrow})
    else
      Nx.reshape(tensor, {nrow, num_classes})
    end
  end

  # ---------- params encoding ----------

  defp encode_params(params) when is_map(params), do: encode_params(Map.to_list(params))

  defp encode_params(params) when is_list(params) do
    params
    |> Enum.map(fn {k, v} -> "#{k}=#{format_param_value(v)}" end)
    |> Enum.join(" ")
  end

  defp format_param_value(true), do: "true"
  defp format_param_value(false), do: "false"
  defp format_param_value(v) when is_atom(v), do: Atom.to_string(v)
  defp format_param_value(v) when is_list(v), do: Enum.map_join(v, ",", &format_param_value/1)
  defp format_param_value(v), do: to_string(v)

  defp maybe_put_categorical_feature(params, []), do: params

  defp maybe_put_categorical_feature(params, categorical_indices) when is_map(params) do
    if param_has_key?(Map.to_list(params), "categorical_feature") do
      params
    else
      Map.put(params, :categorical_feature, categorical_indices)
    end
  end

  defp maybe_put_categorical_feature(params, categorical_indices) when is_list(params) do
    if param_has_key?(params, "categorical_feature") do
      params
    else
      params ++ [categorical_feature: categorical_indices]
    end
  end

  defp param_has_key?(params, key) do
    Enum.any?(params, fn {param_key, _value} -> to_string(param_key) == key end)
  end

  defp importance_type!(:split), do: {0, :s64}
  defp importance_type!(:gain), do: {1, :f64}

  defp importance_type!(other) do
    raise ArgumentError,
          "expected feature importance type to be :gain or :split, got: #{inspect(other)}"
  end

  defp maybe_cast_importance(tensor, :f64), do: tensor
  defp maybe_cast_importance(tensor, :s64), do: Nx.as_type(tensor, :s64)

  defp parse_categorical_features(model_text) do
    model_text
    |> String.split("\n")
    |> Enum.find_value([], fn
      "[categorical_feature: " <> rest ->
        rest
        |> String.trim_trailing("]")
        |> String.trim()
        |> parse_categorical_feature_indexes()

      _line ->
        false
    end)
  end

  defp parse_categorical_feature_indexes(""), do: []

  defp parse_categorical_feature_indexes(value) do
    value
    |> String.split(",", trim: true)
    |> Enum.map(fn index ->
      index
      |> String.trim()
      |> String.to_integer()
    end)
  end

  defp parse_model_params(model_text) do
    model_text
    |> String.split("\n")
    |> Enum.drop_while(&(&1 != "parameters:"))
    |> Enum.drop(1)
    |> Enum.take_while(&(&1 != "end of parameters"))
    |> Enum.reduce(%{}, fn line, acc ->
      case parse_model_param_line(line) do
        nil -> acc
        {key, value} -> Map.put(acc, key, value)
      end
    end)
  end

  defp parse_model_param_line("[" <> rest) do
    rest = String.trim_trailing(rest, "]")

    case String.split(rest, ": ", parts: 2) do
      [key, value] -> {key, parse_model_param_value(value)}
      _other -> nil
    end
  end

  defp parse_model_param_line(_line), do: nil

  defp parse_model_param_value(value) do
    cond do
      String.contains?(value, ",") ->
        value
        |> String.split(",", trim: true)
        |> Enum.map(&parse_model_param_value(String.trim(&1)))

      value == "true" ->
        true

      value == "false" ->
        false

      Regex.match?(~r/^-?\d+$/, value) ->
        String.to_integer(value)

      Regex.match?(~r/^-?\d+\.\d+$/, value) ->
        String.to_float(value)

      true ->
        value
    end
  end

  defp validate_early_stopping_rounds!(rounds) when is_integer(rounds) and rounds > 0, do: rounds

  defp validate_early_stopping_rounds!(rounds) do
    raise ArgumentError,
          "expected :early_stopping_rounds to be a positive integer, got: #{inspect(rounds)}"
  end

  # ---------- NIF result unwrap ----------

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, value}), do: value

  defp unwrap!({:error, msg}) when is_binary(msg) do
    raise RuntimeError, "LightGBM: " <> msg
  end
end
