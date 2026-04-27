defmodule HinokiTest do
  use ExUnit.Case, async: false

  alias Hinoki.{Booster, TestData}

  describe "version/0" do
    test "returns a non-empty binary" do
      v = Hinoki.version()
      assert is_binary(v) and byte_size(v) > 0
    end
  end

  describe "train/2 + predict/2 (Nx tensors)" do
    test "fits a separable binary classifier" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 30,
          params: TestData.deterministic_params()
        )

      assert %Booster{ref: ref} = booster
      assert is_reference(ref)

      preds = Hinoki.predict(booster, features)
      assert Nx.shape(preds) == {Nx.axis_size(features, 0)}

      classes = preds |> Nx.greater(0.5) |> Nx.as_type(:f32)
      accuracy = classes |> Nx.equal(labels) |> Nx.mean() |> Nx.to_number()
      assert accuracy >= 0.95, "expected near-perfect accuracy on separable data, got #{accuracy}"
    end

    test "fits a simple regression model" do
      features =
        Nx.tensor(
          for x <- 0..49 do
            xf = x / 10.0
            [xf, xf * xf]
          end,
          type: :f64
        )

      labels =
        Nx.tensor(
          for x <- 0..49 do
            xf = x / 10.0
            1.5 * xf - 0.25
          end,
          type: :f32
        )

      booster =
        Hinoki.train({features, labels},
          num_iterations: 80,
          params: [
            objective: "regression",
            learning_rate: 0.2,
            min_data_in_leaf: 1,
            num_leaves: 8,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      preds = Hinoki.predict(booster, features)
      assert Nx.shape(preds) == {50}

      mse =
        preds
        |> Nx.subtract(labels)
        |> Nx.pow(2)
        |> Nx.mean()
        |> Nx.to_number()

      assert mse < 0.1
    end

    test "returns a class column for each multiclass class" do
      features =
        Nx.tensor(
          [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [2.0, 0.0],
            [2.1, -0.1],
            [1.9, 0.1]
          ],
          type: :f64
        )

      labels = Nx.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], type: :f32)

      booster =
        Hinoki.train({features, labels},
          num_iterations: 20,
          params: [
            objective: "multiclass",
            num_class: 3,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      preds = Hinoki.predict(booster, features)
      assert Nx.shape(preds) == {9, 3}
    end

    test "stops early when validation stops improving" do
      train_features =
        Nx.tensor(
          for x <- 0..29 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      train_labels =
        Nx.tensor(
          for x <- 0..29 do
            xf = x / 10.0
            :math.sin(xf * 3.0)
          end,
          type: :f32
        )

      valid_features =
        Nx.tensor(
          for x <- 30..59 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      valid_labels = Nx.broadcast(0.0, {30})

      booster =
        Hinoki.train({train_features, train_labels},
          valid: {valid_features, valid_labels},
          early_stopping_rounds: 5,
          num_iterations: 80,
          params: [
            objective: "regression",
            metric: "l2",
            learning_rate: 0.4,
            min_data_in_leaf: 1,
            num_leaves: 16,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      assert Hinoki.current_iteration(booster) < 80

      assert %{
               iteration: best_iteration,
               score: best_score,
               metric: "l2",
               history: history
             } = Hinoki.best(booster)

      assert best_iteration == Hinoki.current_iteration(booster)
      assert is_float(best_score)
      assert Hinoki.info(booster, :best) == Hinoki.best(booster)

      assert history != []
      assert Enum.all?(history, &is_float/1)
      assert Enum.min(history) == best_score
    end

    test "has no best metadata without early stopping" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      assert Hinoki.best(booster) == nil
      assert Hinoki.info(booster, :best) == nil
    end
  end

  describe "save / load round-trip" do
    test "loaded booster reproduces predictions" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      original = Hinoki.predict(booster, features)

      tmp =
        Path.join(System.tmp_dir!(), "hinoki_roundtrip_#{System.unique_integer([:positive])}.txt")

      try do
        :ok = Hinoki.save(booster, tmp)
        loaded = Hinoki.load(tmp)
        again = Hinoki.predict(loaded, features)
        assert Nx.to_flat_list(original) == Nx.to_flat_list(again)
      after
        File.rm(tmp)
      end
    end

    test "directory save preserves Hinoki metadata" do
      train_features =
        Nx.tensor(
          for x <- 0..29 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      train_labels =
        Nx.tensor(
          for x <- 0..29 do
            xf = x / 10.0
            :math.sin(xf * 3.0)
          end,
          type: :f32
        )

      valid_features =
        Nx.tensor(
          for x <- 30..59 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      valid_labels = Nx.broadcast(0.0, {30})

      booster =
        Hinoki.train({train_features, train_labels},
          valid: {valid_features, valid_labels},
          early_stopping_rounds: 5,
          num_iterations: 80,
          params: [
            objective: "regression",
            metric: "l2",
            learning_rate: 0.4,
            min_data_in_leaf: 1,
            num_leaves: 16,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      dir = Path.join(System.tmp_dir!(), "hinoki_bundle_#{System.unique_integer([:positive])}")

      try do
        :ok = Hinoki.save(booster, dir)

        assert File.regular?(Path.join(dir, "model.txt"))
        assert File.regular?(Path.join(dir, "hinoki.json"))

        loaded = Hinoki.load(dir)

        assert Hinoki.best(loaded) == Hinoki.best(booster)

        assert Nx.to_flat_list(Hinoki.predict(loaded, train_features)) ==
                 Nx.to_flat_list(Hinoki.predict(booster, train_features))
      after
        File.rm_rf(dir)
      end
    end

    test "dump/1 + load_string/1 round-trips" do
      {features, labels} = TestData.binary_xor_like(20)

      booster =
        Hinoki.train({features, labels},
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      bin = Hinoki.dump(booster)
      assert is_binary(bin) and byte_size(bin) > 0
      loaded = Hinoki.load_string(bin)
      assert %Booster{} = loaded

      assert Nx.to_flat_list(Hinoki.predict(booster, features)) ==
               Nx.to_flat_list(Hinoki.predict(loaded, features))
    end
  end

  describe "booster introspection" do
    test "returns scalar metadata" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      assert Hinoki.info(booster, :num_features) == 2
      assert Hinoki.num_features(booster) == 2
      assert is_integer(Hinoki.num_classes(booster))
      assert Hinoki.num_classes(booster) >= 1
      assert Hinoki.current_iteration(booster) > 0
      assert %{} = params = Hinoki.info(booster, :params)
      assert params["objective"] == "binary"
      assert params["metric"] == "binary_logloss"
      assert Hinoki.categorical_features(booster) == []
      assert Hinoki.info(booster, :categorical_features) == []
    end

    test "returns feature importance as tensors" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      gain = Hinoki.feature_importance(booster)
      split = Hinoki.feature_importance(booster, :split)

      assert %Nx.Tensor{} = gain
      assert %Nx.Tensor{} = split
      assert Nx.shape(gain) == {2}
      assert Nx.shape(split) == {2}
      assert Nx.type(gain) == {:f, 64}
      assert Nx.type(split) == {:s, 64}
      assert Hinoki.info(booster, :feature_importance) == gain
      assert Hinoki.info(booster, {:feature_importance, :split}) == split
    end

    test "returns feature importance paired with feature names" do
      {features, labels} = TestData.binary_xor_like()

      booster =
        Hinoki.train({features, labels},
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      gain = Hinoki.named_feature_importance(booster, [:left, :right])
      split = Hinoki.named_feature_importance(booster, ["left", "right"], :split)

      assert [{:left, left_gain}, {:right, right_gain}] = gain
      assert [{"left", left_split}, {"right", right_split}] = split
      assert is_float(left_gain)
      assert is_float(right_gain)
      assert is_integer(left_split)
      assert is_integer(right_split)
    end

    test "returns permutation importance score statistics for tensor features" do
      features =
        Nx.tensor(
          for x <- 0..39 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      labels =
        Nx.tensor(
          for x <- 0..39 do
            xf = x / 10.0
            :math.sin(xf)
          end,
          type: :f32
        )

      booster =
        Hinoki.train({features, labels},
          num_iterations: 30,
          params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42, verbose: -1]
        )

      metric_fn = fn y_true, y_pred ->
        y_true
        |> Nx.subtract(y_pred)
        |> Nx.pow(2)
        |> Nx.mean()
      end

      result =
        Hinoki.permutation_importance(booster, features, labels, metric_fn,
          features: [1],
          n_repeats: 3,
          seed: 42
        )

      assert %{baseline_score: baseline, results: [{1, stats}]} = result
      assert is_number(baseline)
      assert %{delta: delta, mean: mean, std: std, scores: scores} = stats
      assert is_number(delta)
      assert is_number(mean)
      assert is_number(std)
      assert length(scores) == 3
      assert Enum.all?(scores, &is_number/1)
      assert_in_delta delta, mean - baseline, 1.0e-12
    end

    test "uses DataFrame column names for permutation importance" do
      df =
        Explorer.DataFrame.new(
          x1: Enum.map(0..39, &(&1 / 10.0)),
          x2: Enum.map(0..39, &:math.sin(&1 / 10.0)),
          y: Enum.map(0..39, &:math.sin(&1 / 10.0))
        )

      booster =
        Hinoki.train(df,
          target: :y,
          num_iterations: 30,
          params: [objective: "regression", metric: "l2", num_threads: 1, seed: 42, verbose: -1]
        )

      x = Explorer.DataFrame.discard(df, ["y"])

      y =
        df
        |> Explorer.DataFrame.pull("y")
        |> Explorer.Series.to_tensor()

      metric_fn = fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end

      result =
        Hinoki.permutation_importance(booster, x, y, metric_fn,
          features: [:x2],
          n_repeats: 2,
          seed: 42
        )

      assert %{results: [{"x2", %{scores: scores}}]} = result
      assert length(scores) == 2
    end

    test "rejects unsupported info and importance types" do
      {features, labels} = TestData.binary_xor_like(10)

      booster =
        Hinoki.train({features, labels},
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      assert_raise ArgumentError, ~r/unsupported booster info key/, fn ->
        Hinoki.info(booster, :unknown)
      end

      assert_raise ArgumentError, ~r/expected feature importance type/, fn ->
        Hinoki.feature_importance(booster, :weight)
      end

      assert_raise ArgumentError, ~r/expected 2 feature names/, fn ->
        Hinoki.named_feature_importance(booster, [:only_one])
      end
    end

    test "rejects invalid permutation importance options" do
      {features, labels} = TestData.binary_xor_like(10)

      booster =
        Hinoki.train({features, labels},
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      metric_fn = fn _y_true, _y_pred -> 0.0 end

      assert_raise ArgumentError, ~r/expected :n_repeats/, fn ->
        Hinoki.permutation_importance(booster, features, labels, metric_fn, n_repeats: 0)
      end

      assert_raise ArgumentError, ~r/permutation feature 2 not found/, fn ->
        Hinoki.permutation_importance(booster, features, labels, metric_fn, features: [2])
      end
    end
  end

  describe "Explorer.DataFrame input" do
    test "trains from a DataFrame with :target option" do
      df =
        Explorer.DataFrame.new(
          x1: [0.0, 0.1, -0.1, 0.9, 1.0, 1.1],
          x2: [0.0, -0.1, 0.1, 1.0, 0.9, 1.1],
          y: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        )

      booster =
        Hinoki.train(df,
          target: :y,
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      preds = Hinoki.predict(booster, Explorer.DataFrame.discard(df, ["y"]))
      assert Nx.shape(preds) == {6}
    end

    test "marks category columns as LightGBM categorical features" do
      df =
        Explorer.DataFrame.new(
          group:
            Explorer.Series.from_list(
              ["low", "low", "mid", "mid", "high", "high", "low", "high"],
              dtype: :category
            ),
          x: Explorer.Series.from_list([0.1, 0.2, 1.0, 1.1, 2.0, 2.1, 0.3, 2.2]),
          y: Explorer.Series.from_list([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        )

      booster =
        Hinoki.train(df,
          target: :y,
          num_iterations: 20,
          params: TestData.deterministic_params()
        )

      preds = Hinoki.predict(booster, Explorer.DataFrame.discard(df, ["y"]))

      assert Nx.shape(preds) == {8}
      assert Hinoki.num_features(booster) == 2
      assert Hinoki.categorical_features(booster) == [0]
      assert Hinoki.info(booster, :categorical_features) == [0]
      assert Hinoki.dump(booster) =~ "[categorical_feature: 0]"
    end
  end

  describe "Hinoki.CV.k_fold/2" do
    test "runs tensor k-fold cross-validation with early stopping best results" do
      features =
        Nx.tensor(
          for x <- 0..59 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      labels =
        Nx.tensor(
          for x <- 0..59 do
            xf = x / 10.0
            :math.sin(xf * 3.0)
          end,
          type: :f32
        )

      result =
        Hinoki.CV.k_fold({features, labels},
          k: 3,
          max_concurrency: 2,
          early_stopping_rounds: 5,
          num_iterations: 40,
          params: [
            objective: "regression",
            metric: "l2",
            learning_rate: 0.3,
            min_data_in_leaf: 1,
            num_leaves: 16,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      assert %{folds: folds, stats: stats} = result
      assert length(folds) == 3
      assert Enum.all?(folds, &match?(%{iteration: _, score: _, metric: "l2"}, &1))
      assert Enum.all?(folds, &is_integer(&1.iteration))
      assert Enum.all?(folds, &is_float(&1.score))

      assert %{
               metric: "l2",
               score: %{mean: score_mean, std: score_std},
               iteration: %{mean: iteration_mean, std: iteration_std}
             } = stats

      assert is_float(score_mean)
      assert is_float(score_std)
      assert is_float(iteration_mean)
      assert is_float(iteration_std)
    end

    test "requires early stopping and valid k" do
      {features, labels} = TestData.binary_xor_like(3)

      assert_raise ArgumentError, ~r/requires :early_stopping_rounds/, fn ->
        Hinoki.CV.k_fold({features, labels}, k: 2, params: TestData.deterministic_params())
      end

      assert_raise ArgumentError, ~r/expected :k/, fn ->
        Hinoki.CV.k_fold({features, labels},
          k: 1,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end

      assert_raise ArgumentError, ~r/expected :max_concurrency/, fn ->
        Hinoki.CV.k_fold({features, labels},
          k: 2,
          max_concurrency: 0,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end

      assert_raise ArgumentError, ~r/expected :folding_rule/, fn ->
        Hinoki.CV.k_fold({features, labels},
          k: 2,
          folding_rule: :unknown,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end

      assert_raise ArgumentError, ~r/expected :seed/, fn ->
        Hinoki.CV.k_fold({features, labels},
          k: 2,
          folding_rule: :shuffle,
          seed: "42",
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end
    end

    test "supports shuffled tensor k-fold cross-validation" do
      {features, labels} = TestData.binary_xor_like(30)

      result =
        Hinoki.CV.k_fold({features, labels},
          k: 3,
          folding_rule: :shuffle,
          seed: 42,
          early_stopping_rounds: 3,
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      assert %{folds: folds, stats: %{metric: "binary_logloss"}} = result
      assert length(folds) == 3
    end

    test "seeded shuffle does not overwrite caller random state" do
      {features, labels} = TestData.binary_xor_like(30)

      :rand.seed(:exsss, {1, 2, 3})
      expected = :rand.uniform()

      :rand.seed(:exsss, {1, 2, 3})

      Hinoki.CV.k_fold({features, labels},
        k: 3,
        folding_rule: :shuffle,
        seed: 42,
        early_stopping_rounds: 3,
        num_iterations: 10,
        params: TestData.deterministic_params()
      )

      assert :rand.uniform() == expected
    end

    test "stratified k-fold requires enough rows per label group" do
      features = Nx.tensor([[0.0], [1.0], [2.0], [3.0]], type: :f32)
      labels = Nx.tensor([0.0, 1.0, 2.0, 3.0], type: :f32)

      assert_raise ArgumentError, ~r/every label group to have at least k rows/, fn ->
        Hinoki.CV.k_fold({features, labels},
          k: 2,
          folding_rule: :stratified,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end
    end

    test "runs DataFrame k-fold cross-validation with target option" do
      df =
        Explorer.DataFrame.new(
          x1:
            for x <- 0..59 do
              x / 10.0
            end,
          x2:
            for x <- 0..59 do
              :math.sin(x / 10.0)
            end,
          y:
            for x <- 0..59 do
              :math.sin(x / 10.0 * 3.0)
            end
        )

      result =
        Hinoki.CV.k_fold(df,
          target: :y,
          k: 3,
          early_stopping_rounds: 5,
          num_iterations: 40,
          params: [
            objective: "regression",
            metric: "l2",
            learning_rate: 0.3,
            min_data_in_leaf: 1,
            num_leaves: 16,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      assert %{folds: folds, stats: %{metric: "l2"}} = result
      assert length(folds) == 3
      assert Enum.all?(folds, &match?(%{iteration: _, score: _, metric: "l2"}, &1))
    end

    test "supports stratified shuffled DataFrame k-fold cross-validation" do
      df =
        Explorer.DataFrame.new(
          x1: Enum.map(0..59, &(&1 / 10.0)),
          x2: Enum.map(0..59, &:math.sin(&1 / 10.0)),
          y: Enum.map(0..59, &rem(&1, 2))
        )

      result =
        Hinoki.CV.k_fold(df,
          target: :y,
          k: 3,
          folding_rule: :stratified_shuffle,
          seed: 42,
          early_stopping_rounds: 3,
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      assert %{folds: folds, stats: %{metric: "binary_logloss"}} = result
      assert length(folds) == 3
    end

    test "DataFrame k-fold requires target option" do
      df = Explorer.DataFrame.new(x: [1.0, 2.0, 3.0], y: [0.0, 1.0, 0.0])

      assert_raise ArgumentError, ~r/requires the :target option/, fn ->
        Hinoki.CV.k_fold(df,
          k: 2,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end
    end
  end

  describe "Hinoki.CV.grid_search/3" do
    test "runs tensor k-fold cross-validation for every parameter combination" do
      features =
        Nx.tensor(
          for x <- 0..39 do
            xf = x / 10.0
            [xf, :math.sin(xf)]
          end,
          type: :f64
        )

      labels =
        Nx.tensor(
          for x <- 0..39 do
            xf = x / 10.0
            :math.sin(xf * 3.0)
          end,
          type: :f32
        )

      result =
        Hinoki.CV.grid_search(
          {features, labels},
          [learning_rate: [0.2, 0.3], num_leaves: [8, 16]],
          k: 2,
          max_concurrency: 2,
          early_stopping_rounds: 3,
          num_iterations: 20,
          params: [
            objective: "regression",
            metric: "l2",
            learning_rate: 0.1,
            min_data_in_leaf: 1,
            num_leaves: 4,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      assert %{results: results} = result
      assert length(results) == 4

      assert Enum.map(results, fn %{params: params} ->
               {params[:learning_rate], params[:num_leaves]}
             end) == [
               {0.2, 8},
               {0.2, 16},
               {0.3, 8},
               {0.3, 16}
             ]

      assert Enum.all?(results, &match?(%{cv: %{folds: [_, _], stats: %{metric: "l2"}}}, &1))
    end

    test "runs DataFrame grid search with target option" do
      df =
        Explorer.DataFrame.new(
          x:
            for x <- 0..39 do
              x / 10.0
            end,
          y:
            for x <- 0..39 do
              :math.sin(x / 10.0)
            end
        )

      result =
        Hinoki.CV.grid_search(df, [learning_rate: [0.2, 0.3]],
          target: :y,
          k: 2,
          early_stopping_rounds: 3,
          num_iterations: 20,
          params: [
            objective: "regression",
            metric: "l2",
            min_data_in_leaf: 1,
            num_leaves: 8,
            num_threads: 1,
            seed: 42,
            verbose: -1
          ]
        )

      assert %{results: [%{cv: %{stats: %{metric: "l2"}}}, %{cv: %{stats: %{metric: "l2"}}}]} =
               result
    end

    test "requires non-empty grid value lists" do
      {features, labels} = TestData.binary_xor_like(4)

      assert_raise ArgumentError, ~r/non-empty list/, fn ->
        Hinoki.CV.grid_search({features, labels}, [learning_rate: []],
          k: 2,
          early_stopping_rounds: 2,
          params: TestData.deterministic_params()
        )
      end
    end
  end

  describe "input validation" do
    test "labels row count must match features" do
      features = Nx.iota({10, 3}, type: :f64)
      labels = Nx.iota({9}, type: :f32)

      assert_raise ArgumentError, ~r/labels tensor shape \{9\}/, fn ->
        Hinoki.train({features, labels}, params: TestData.deterministic_params())
      end
    end

    test "feature tensor must be 2D" do
      features = Nx.iota({10}, type: :f64)
      labels = Nx.iota({10}, type: :f32)

      assert_raise ArgumentError, ~r/expected 2D feature tensor/, fn ->
        Hinoki.train({features, labels}, params: TestData.deterministic_params())
      end
    end

    test "DataFrame requires :target" do
      df = Explorer.DataFrame.new(x: [1.0, 2.0], y: [0.0, 1.0])

      assert_raise ArgumentError, ~r/:target option/, fn ->
        Hinoki.train(df, params: TestData.deterministic_params())
      end
    end

    test "missing target column raises with available column list" do
      df = Explorer.DataFrame.new(x: [1.0, 2.0], y: [0.0, 1.0])

      assert_raise ArgumentError, ~r/target column "z" not found/, fn ->
        Hinoki.train(df, target: :z, params: TestData.deterministic_params())
      end
    end

    test "prediction feature count must match the trained booster" do
      {features, labels} = TestData.binary_xor_like(10)

      booster =
        Hinoki.train({features, labels},
          num_iterations: 10,
          params: TestData.deterministic_params()
        )

      wrong_features = Nx.iota({20, 3}, type: :f64)

      assert_raise RuntimeError, ~r/^LightGBM:/, fn ->
        Hinoki.predict(booster, wrong_features)
      end
    end

    test "early stopping requires validation data" do
      {features, labels} = TestData.binary_xor_like(10)

      assert_raise ArgumentError, ~r/:early_stopping_rounds requires/, fn ->
        Hinoki.train({features, labels},
          early_stopping_rounds: 5,
          params: TestData.deterministic_params()
        )
      end
    end

    test "early stopping rounds must be positive" do
      {features, labels} = TestData.binary_xor_like(10)

      assert_raise ArgumentError, ~r/positive integer/, fn ->
        Hinoki.train({features, labels},
          valid: {features, labels},
          early_stopping_rounds: 0,
          params: TestData.deterministic_params()
        )
      end
    end

    test "validation feature count must match training features" do
      {features, labels} = TestData.binary_xor_like(10)
      valid_features = Nx.iota({20, 3}, type: :f64)

      assert_raise ArgumentError, ~r/validation feature count 3/, fn ->
        Hinoki.train({features, labels},
          valid: {valid_features, labels},
          early_stopping_rounds: 5,
          params: TestData.deterministic_params()
        )
      end
    end
  end

  describe "NIF error surface" do
    test "load_string/1 on garbage raises RuntimeError prefixed with LightGBM" do
      assert_raise RuntimeError, ~r/^LightGBM:/, fn ->
        Hinoki.load_string("not a real lightgbm model")
      end
    end
  end
end
