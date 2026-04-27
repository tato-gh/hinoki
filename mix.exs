defmodule Hinoki.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/tsukinolab/hinoki"

  def project do
    [
      app: :hinoki,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      name: "Hinoki",
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:nx, "~> 0.7"},
      {:explorer, "~> 0.8"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  defp description do
    "Hinoki — a LightGBM binding for Elixir. Japanese cypress, a famously lightweight, durable wood."
  end

  defp package do
    [
      maintainers: ["tsukinolab"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib c_src Makefile mix.exs README.md LICENSE .formatter.exs)
    ]
  end

  defp docs do
    [
      main: "Hinoki",
      extras: ["README.md"]
    ]
  end
end
