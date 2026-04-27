defmodule Hinoki.Booster do
  @moduledoc """
  An opaque handle to a trained LightGBM booster.

  The struct holds a NIF resource reference plus optional training-session
  evaluation metadata. Booster properties such as the number of features or
  classes are recovered from the native booster on demand rather than cached.
  """

  @type best :: %{
          iteration: non_neg_integer(),
          score: float(),
          metric: String.t()
        }

  @type evals_result :: %{optional(String.t()) => %{optional(String.t()) => [float()]}}

  @type t :: %__MODULE__{
          ref: reference(),
          best: best() | nil,
          evals_result: evals_result()
        }

  defstruct [:ref, :best, evals_result: %{}]
end
