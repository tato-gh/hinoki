defmodule Hinoki.Booster do
  @moduledoc """
  An opaque handle to a trained LightGBM booster.

  The struct holds only a NIF resource reference; metadata such as the
  number of features or classes is recovered from the booster on demand
  rather than cached on the struct.
  """

  @type t :: %__MODULE__{ref: reference()}
  defstruct [:ref]
end
