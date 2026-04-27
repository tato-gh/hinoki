defmodule Hinoki.Booster do
  @moduledoc """
  An opaque handle to a trained LightGBM booster.

  The struct holds a NIF resource reference plus optional early-stopping
  metadata. Booster properties such as the number of features or classes
  are recovered from the native booster on demand rather than cached.
  """

  @type best :: %{
          iteration: non_neg_integer(),
          score: float(),
          metric: String.t(),
          history: [float()]
        }

  @type t :: %__MODULE__{
          ref: reference(),
          best: best() | nil
        }

  defstruct [:ref, :best]
end
