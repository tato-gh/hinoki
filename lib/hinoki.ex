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
    * `:valid` — validation data for early stopping. Pass a `{features, labels}`
      tuple, a DataFrame, or a list of either. DataFrame validation uses the same
      `:target` column as training.
    * `:early_stopping_rounds` — stop training when the first metric on the first
      validation dataset does not improve for this many rounds.
  """
  @spec train(term(), keyword()) :: Booster.t()
  def train(input, opts \\ []) do
    num_iter = Keyword.get(opts, :num_iterations, @default_num_iterations)
    params = Keyword.get(opts, :params, [])
    target = Keyword.get(opts, :target)
    early_stopping_rounds = Keyword.get(opts, :early_stopping_rounds)

    {features_bin, labels_bin, nrow, ncol, categorical_indices} = to_train_payload(input, target)
    valid_payloads = to_valid_payloads(Keyword.get(opts, :valid, []), input, target, ncol)
    params = maybe_put_categorical_feature(params, categorical_indices)
    params_bin = encode_params(params)

    dataset_ref = create_dataset!(features_bin, labels_bin, nrow, ncol, params_bin)

    booster_ref = unwrap!(NIF.booster_create(dataset_ref, params_bin))

    valid_refs =
      Enum.map(valid_payloads, fn {valid_features_bin, valid_labels_bin, valid_nrow, valid_ncol} ->
        valid_ref =
          create_dataset!(
            valid_features_bin,
            valid_labels_bin,
            valid_nrow,
            valid_ncol,
            params_bin,
            dataset_ref
          )

        unwrap!(NIF.booster_add_valid_data(booster_ref, valid_ref))
        valid_ref
      end)

    training_metadata =
      run_training!(booster_ref, num_iter, early_stopping_rounds, valid_payloads, valid_refs)

    struct!(Booster, Keyword.put(training_metadata, :ref, booster_ref))
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
  Hinoki metadata such as `best` and `evals_result`. When saving to a new path,
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
    * `:best`
    * `:evals_result`
    * `:categorical_features`
    * `:feature_importance` — equivalent to `{:feature_importance, :gain}`
    * `{:feature_importance, :gain}`
    * `{:feature_importance, :split}`
  """
  @spec info(Booster.t(), atom() | tuple()) ::
          integer()
          | float()
          | nil
          | [non_neg_integer()]
          | Booster.best()
          | Booster.evals_result()
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

  def info(%Booster{} = booster, :best) do
    best(booster)
  end

  def info(%Booster{evals_result: evals_result}, :evals_result) do
    evals_result
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

  # ---------- input → binary ----------

  defp create_dataset!(features_bin, labels_bin, nrow, ncol, params_bin, reference_ref \\ nil) do
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
    dataset_ref
  end

  defp run_training!(booster_ref, num_iter, nil, _valid_payloads, _valid_refs) do
    unwrap!(NIF.booster_update_iters(booster_ref, num_iter))
    [best: nil, evals_result: %{}]
  end

  defp run_training!(booster_ref, num_iter, early_stopping_rounds, valid_payloads, valid_refs) do
    early_stopping_rounds = validate_early_stopping_rounds!(early_stopping_rounds)

    if valid_payloads == [] do
      raise ArgumentError,
            ":early_stopping_rounds requires at least one validation dataset in :valid"
    end

    _ = valid_refs

    {best_iteration, best_score, metric_name, scores} =
      unwrap!(
        NIF.booster_update_iters_early_stopping(booster_ref, num_iter, early_stopping_rounds)
      )

    dataset_name = "valid_0"

    [
      best: %{
        iteration: best_iteration,
        score: best_score,
        metric: metric_name
      },
      evals_result: %{dataset_name => %{metric_name => scores}}
    ]
  end

  defp bundle_path?(path) do
    File.dir?(path) or
      (not File.exists?(path) and Path.extname(path) == "")
  end

  defp save_bundle(%Booster{} = booster, path) do
    File.mkdir_p!(path)
    File.write!(Path.join(path, @bundle_model_filename), dump(booster))

    metadata = %{
      "best" => encode_best(booster.best),
      "evals_result" => booster.evals_result
    }

    File.write!(Path.join(path, @bundle_metadata_filename), :json.encode(metadata))
  end

  defp load_bundle(path) do
    booster = Path.join(path, @bundle_model_filename) |> File.read!() |> load_string()

    metadata =
      path
      |> Path.join(@bundle_metadata_filename)
      |> File.read!()
      |> :json.decode()

    %Booster{
      booster
      | best: decode_best(Map.get(metadata, "best")),
        evals_result: Map.get(metadata, "evals_result", %{})
    }
  end

  defp encode_best(nil), do: nil

  defp encode_best(%{iteration: iteration, score: score, metric: metric}) do
    %{
      "iteration" => iteration,
      "score" => score,
      "metric" => metric
    }
  end

  defp decode_best(nil), do: nil

  defp decode_best(%{"iteration" => iteration, "score" => score, "metric" => metric}) do
    %{
      iteration: iteration,
      score: score,
      metric: metric
    }
  end

  defp to_train_payload({%Nx.Tensor{} = features, %Nx.Tensor{} = labels}, _target) do
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    labels_bin = labels_tensor_to_bin(labels, nrow)
    {features_bin, labels_bin, nrow, ncol, []}
  end

  defp to_train_payload(%Explorer.DataFrame{} = df, target)
       when (is_atom(target) and not is_nil(target)) or is_binary(target) do
    target = to_string(target)
    names = Explorer.DataFrame.names(df)

    unless target in names do
      raise ArgumentError,
            "target column #{inspect(target)} not found in DataFrame; available columns: #{inspect(names)}"
    end

    feature_cols = names -- [target]
    dtypes = Explorer.DataFrame.dtypes(df)

    if feature_cols == [] do
      raise ArgumentError,
            "DataFrame has no feature columns after dropping target #{inspect(target)}"
    end

    features = df_columns_to_tensor(df, feature_cols)
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    categorical_indices = categorical_feature_indices(feature_cols, dtypes)

    labels =
      df
      |> Explorer.DataFrame.pull(target)
      |> Explorer.Series.cast({:f, 32})
      |> Explorer.Series.to_tensor()

    labels_bin = labels_tensor_to_bin(labels, nrow)
    {features_bin, labels_bin, nrow, ncol, categorical_indices}
  end

  defp to_train_payload(%Explorer.DataFrame{}, nil) do
    raise ArgumentError,
          "training from a DataFrame requires the :target option naming the label column"
  end

  defp to_train_payload(other, _target) do
    raise ArgumentError,
          "expected an Explorer.DataFrame or {features, labels} tensor tuple, got: #{inspect(other)}"
  end

  defp to_valid_payloads(nil, _train_input, _target, _expected_ncol), do: []
  defp to_valid_payloads([], _train_input, _target, _expected_ncol), do: []

  defp to_valid_payloads(valid, train_input, target, expected_ncol) do
    valid
    |> normalize_valid()
    |> Enum.map(&to_valid_payload(&1, train_input, target, expected_ncol))
  end

  defp normalize_valid({%Nx.Tensor{}, %Nx.Tensor{}} = valid), do: [valid]
  defp normalize_valid(%Explorer.DataFrame{} = valid), do: [valid]
  defp normalize_valid(valid) when is_list(valid), do: valid

  defp normalize_valid(other) do
    raise ArgumentError,
          "expected :valid to be a validation dataset or a list of validation datasets, got: #{inspect(other)}"
  end

  defp to_valid_payload(
         {%Nx.Tensor{} = features, %Nx.Tensor{} = labels},
         _train_input,
         _target,
         expected_ncol
       ) do
    {features_bin, nrow, ncol} = tensor_to_features_bin(features)
    validate_feature_count!(ncol, expected_ncol, "validation")
    labels_bin = labels_tensor_to_bin(labels, nrow)
    {features_bin, labels_bin, nrow, ncol}
  end

  defp to_valid_payload(%Explorer.DataFrame{} = df, %Explorer.DataFrame{}, target, expected_ncol) do
    {features_bin, labels_bin, nrow, ncol, _categorical_indices} = to_train_payload(df, target)
    validate_feature_count!(ncol, expected_ncol, "validation")
    {features_bin, labels_bin, nrow, ncol}
  end

  defp to_valid_payload(%Explorer.DataFrame{}, _train_input, _target, _expected_ncol) do
    raise ArgumentError,
          "DataFrame validation data is only supported when training input is a DataFrame"
  end

  defp to_valid_payload(other, _train_input, _target, _expected_ncol) do
    raise ArgumentError,
          "expected each validation dataset to be an Explorer.DataFrame or {features, labels} tensor tuple, got: #{inspect(other)}"
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

  defp validate_feature_count!(ncol, ncol, _context), do: :ok

  defp validate_feature_count!(ncol, expected_ncol, context) do
    raise ArgumentError,
          "#{context} feature count #{ncol} does not match training feature count #{expected_ncol}"
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
