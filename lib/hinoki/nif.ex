defmodule Hinoki.NIF do
  @moduledoc false

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:hinoki), ~c"libhinoki_nif")
    :erlang.load_nif(path, 0)
  end

  def lgbm_version, do: :erlang.nif_error(:nif_not_loaded)

  def dataset_create_from_mat(_features_bin, _nrow, _ncol, _params_bin),
    do: :erlang.nif_error(:nif_not_loaded)

  def dataset_set_label(_dataset_ref, _labels_bin),
    do: :erlang.nif_error(:nif_not_loaded)

  def booster_create(_dataset_ref, _params_bin),
    do: :erlang.nif_error(:nif_not_loaded)

  def booster_update_iters(_booster_ref, _n),
    do: :erlang.nif_error(:nif_not_loaded)

  def booster_predict_for_mat(_booster_ref, _features_bin, _nrow, _ncol, _params_bin),
    do: :erlang.nif_error(:nif_not_loaded)

  def booster_save_model_to_string(_booster_ref, _start_iter, _num_iter, _importance_type),
    do: :erlang.nif_error(:nif_not_loaded)

  def booster_load_model_from_string(_model_bin),
    do: :erlang.nif_error(:nif_not_loaded)
end
