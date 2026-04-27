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
