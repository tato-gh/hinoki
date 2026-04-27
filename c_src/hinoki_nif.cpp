// Hinoki NIF — minimal LightGBM C-API wrapper for Elixir.
//
// All heavy calls are scheduled as dirty NIFs.
// Resources hold raw LightGBM handles and free them in their dtor,
// so process death cleans up via BEAM GC.

#include <erl_nif.h>
#include <LightGBM/c_api.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>
#include <limits>

static ErlNifResourceType *HINOKI_DATASET_RES;
static ErlNifResourceType *HINOKI_BOOSTER_RES;

typedef struct {
    DatasetHandle handle;
} HinokiDataset;

typedef struct {
    BoosterHandle handle;
} HinokiBooster;

static void hinoki_dataset_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    HinokiDataset *d = (HinokiDataset *)obj;
    if (d->handle != NULL) {
        LGBM_DatasetFree(d->handle);
        d->handle = NULL;
    }
}

static void hinoki_booster_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    HinokiBooster *b = (HinokiBooster *)obj;
    if (b->handle != NULL) {
        LGBM_BoosterFree(b->handle);
        b->handle = NULL;
    }
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)load_info;

    ErlNifResourceFlags flags =
        (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    HINOKI_DATASET_RES = enif_open_resource_type(
        env, NULL, "Hinoki.Dataset", hinoki_dataset_dtor, flags, NULL);
    HINOKI_BOOSTER_RES = enif_open_resource_type(
        env, NULL, "Hinoki.Booster", hinoki_booster_dtor, flags, NULL);

    if (HINOKI_DATASET_RES == NULL || HINOKI_BOOSTER_RES == NULL) {
        return 1;
    }
    return 0;
}

static int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data,
                   ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)old_priv_data;
    (void)load_info;

    HINOKI_DATASET_RES = enif_open_resource_type(
        env, NULL, "Hinoki.Dataset", hinoki_dataset_dtor,
        ERL_NIF_RT_TAKEOVER, NULL);
    HINOKI_BOOSTER_RES = enif_open_resource_type(
        env, NULL, "Hinoki.Booster", hinoki_booster_dtor,
        ERL_NIF_RT_TAKEOVER, NULL);
    if (HINOKI_DATASET_RES == NULL || HINOKI_BOOSTER_RES == NULL) return 1;
    return 0;
}

// ---------- helpers ----------

static ERL_NIF_TERM mk_atom(ErlNifEnv *env, const char *s) {
    ERL_NIF_TERM a;
    if (enif_make_existing_atom(env, s, &a, ERL_NIF_LATIN1)) return a;
    return enif_make_atom(env, s);
}

static ERL_NIF_TERM mk_binary(ErlNifEnv *env, const char *s, size_t len) {
    ERL_NIF_TERM bin;
    unsigned char *p = enif_make_new_binary(env, len, &bin);
    if (len > 0) memcpy(p, s, len);
    return bin;
}

static ERL_NIF_TERM mk_ok(ErlNifEnv *env, ERL_NIF_TERM v) {
    return enif_make_tuple2(env, mk_atom(env, "ok"), v);
}

static ERL_NIF_TERM mk_error(ErlNifEnv *env, const char *msg) {
    return enif_make_tuple2(env, mk_atom(env, "error"),
                            mk_binary(env, msg, strlen(msg)));
}

static ERL_NIF_TERM mk_lgbm_error(ErlNifEnv *env) {
    const char *err = LGBM_GetLastError();
    return mk_error(env, err != NULL ? err : "unknown LightGBM error");
}

// Copy a binary term into a freshly-allocated NUL-terminated C string.
// Caller must free.
static char *binary_to_cstring(ErlNifEnv *env, ERL_NIF_TERM term) {
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, term, &bin)) return NULL;
    char *s = (char *)enif_alloc(bin.size + 1);
    if (s == NULL) return NULL;
    if (bin.size > 0) memcpy(s, bin.data, bin.size);
    s[bin.size] = '\0';
    return s;
}

static bool higher_is_better(const std::string &name) {
    return name.rfind("auc", 0) == 0 ||
           name.rfind("ndcg@", 0) == 0 ||
           name.rfind("map@", 0) == 0 ||
           name.rfind("average_precision", 0) == 0;
}

static int get_first_eval_name(BoosterHandle handle, std::string *out) {
    int eval_count = 0;
    int rc = LGBM_BoosterGetEvalCounts(handle, &eval_count);
    if (rc != 0) return rc;
    if (eval_count <= 0) return -2;

    const size_t buffer_len = 256;
    size_t out_buffer_len = 0;
    int out_len = 0;
    std::vector<std::vector<char> > buffers(eval_count, std::vector<char>(buffer_len, '\0'));
    std::vector<char *> names(eval_count, NULL);
    for (int i = 0; i < eval_count; i++) names[i] = buffers[i].data();

    rc = LGBM_BoosterGetEvalNames(handle, eval_count, &out_len, buffer_len,
                                  &out_buffer_len, names.data());
    if (rc != 0) return rc;
    if (out_buffer_len > buffer_len) {
        for (int i = 0; i < eval_count; i++) {
            buffers[i].assign(out_buffer_len, '\0');
            names[i] = buffers[i].data();
        }
        rc = LGBM_BoosterGetEvalNames(handle, eval_count, &out_len,
                                      out_buffer_len, &out_buffer_len,
                                      names.data());
        if (rc != 0) return rc;
    }

    *out = std::string(buffers[0].data());
    return 0;
}

static int get_first_valid_eval(BoosterHandle handle, double *out) {
    int eval_count = 0;
    int rc = LGBM_BoosterGetEvalCounts(handle, &eval_count);
    if (rc != 0) return rc;
    if (eval_count <= 0) return -2;

    std::vector<double> results(eval_count, 0.0);
    int out_len = 0;
    rc = LGBM_BoosterGetEval(handle, 1, &out_len, results.data());
    if (rc != 0) return rc;
    if (out_len <= 0) return -2;

    *out = results[0];
    return 0;
}

// ---------- NIFs ----------

static ERL_NIF_TERM nif_lgbm_version(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    // LightGBM doesn't expose a runtime version string via c_api;
    // use the build-time constant if present.
#ifdef LIGHTGBM_VERSION
    const char *v = LIGHTGBM_VERSION;
#else
    const char *v = "unknown";
#endif
    return mk_binary(env, v, strlen(v));
}

static ERL_NIF_TERM create_dataset_from_mat(ErlNifEnv *env, ERL_NIF_TERM features_term,
                                            ERL_NIF_TERM nrow_term,
                                            ERL_NIF_TERM ncol_term,
                                            ERL_NIF_TERM params_term,
                                            DatasetHandle reference) {
    ErlNifBinary features;
    int nrow, ncol;
    if (!enif_inspect_binary(env, features_term, &features)) return enif_make_badarg(env);
    if (!enif_get_int(env, nrow_term, &nrow)) return enif_make_badarg(env);
    if (!enif_get_int(env, ncol_term, &ncol)) return enif_make_badarg(env);

    char *params = binary_to_cstring(env, params_term);
    if (params == NULL) return enif_make_badarg(env);

    if ((size_t)nrow * (size_t)ncol * sizeof(double) != features.size) {
        enif_free(params);
        return mk_error(env, "feature binary size does not match nrow*ncol*8");
    }

    DatasetHandle out = NULL;
    int rc = LGBM_DatasetCreateFromMat(features.data, C_API_DTYPE_FLOAT64,
                                       nrow, ncol, /*is_row_major=*/1,
                                       params, reference, &out);
    enif_free(params);
    if (rc != 0) return mk_lgbm_error(env);

    HinokiDataset *res = (HinokiDataset *)enif_alloc_resource(
        HINOKI_DATASET_RES, sizeof(HinokiDataset));
    res->handle = out;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return mk_ok(env, term);
}

// dataset_create_from_mat(features_bin, nrow, ncol, params_bin)
//   features_bin: row-major Float64 doubles, length = nrow*ncol*8 bytes
static ERL_NIF_TERM nif_dataset_create_from_mat(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
    (void)argc;
    return create_dataset_from_mat(env, argv[0], argv[1], argv[2], argv[3], NULL);
}

// dataset_create_from_mat_reference(features_bin, nrow, ncol, params_bin, reference)
static ERL_NIF_TERM nif_dataset_create_from_mat_reference(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiDataset *reference;
    if (!enif_get_resource(env, argv[4], HINOKI_DATASET_RES, (void **)&reference))
        return enif_make_badarg(env);

    return create_dataset_from_mat(env, argv[0], argv[1], argv[2], argv[3],
                                   reference->handle);
}

// dataset_set_label(dataset, labels_bin)
//   labels_bin: Float32, length = nrow*4
static ERL_NIF_TERM nif_dataset_set_label(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiDataset *d;
    ErlNifBinary labels;
    if (!enif_get_resource(env, argv[0], HINOKI_DATASET_RES, (void **)&d))
        return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[1], &labels))
        return enif_make_badarg(env);

    int nrow = (int)(labels.size / sizeof(float));
    int rc = LGBM_DatasetSetField(d->handle, "label", labels.data, nrow,
                                  C_API_DTYPE_FLOAT32);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_atom(env, "ok");
}

// booster_create(dataset, params_bin)
static ERL_NIF_TERM nif_booster_create(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiDataset *d;
    if (!enif_get_resource(env, argv[0], HINOKI_DATASET_RES, (void **)&d))
        return enif_make_badarg(env);

    char *params = binary_to_cstring(env, argv[1]);
    if (params == NULL) return enif_make_badarg(env);

    BoosterHandle out = NULL;
    int rc = LGBM_BoosterCreate(d->handle, params, &out);
    enif_free(params);
    if (rc != 0) return mk_lgbm_error(env);

    HinokiBooster *res = (HinokiBooster *)enif_alloc_resource(
        HINOKI_BOOSTER_RES, sizeof(HinokiBooster));
    res->handle = out;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return mk_ok(env, term);
}

// booster_add_valid_data(booster, dataset)
static ERL_NIF_TERM nif_booster_add_valid_data(ErlNifEnv *env, int argc,
                                               const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    HinokiDataset *d;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_get_resource(env, argv[1], HINOKI_DATASET_RES, (void **)&d))
        return enif_make_badarg(env);

    int rc = LGBM_BoosterAddValidData(b->handle, d->handle);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_atom(env, "ok");
}

// booster_update_iters(booster, n)
//   Runs up to n boosting iterations; stops early if LightGBM signals
//   convergence. Returns {:ok, iterations_run}.
static ERL_NIF_TERM nif_booster_update_iters(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    int n;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &n)) return enif_make_badarg(env);

    int ran = 0;
    for (int i = 0; i < n; i++) {
        int finished = 0;
        int rc = LGBM_BoosterUpdateOneIter(b->handle, &finished);
        if (rc != 0) return mk_lgbm_error(env);
        ran++;
        if (finished) break;
    }
    return mk_ok(env, enif_make_int(env, ran));
}

// booster_update_iters_early_stopping(booster, n, stopping_rounds)
//   Monitors the first metric on the first validation dataset.
//   Returns {:ok, {best_iteration, best_score, metric_name, scores}}
//   after rolling back to the best iteration.
static ERL_NIF_TERM nif_booster_update_iters_early_stopping(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    int n, stopping_rounds;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &n)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &stopping_rounds)) return enif_make_badarg(env);

    std::string eval_name;
    int rc = get_first_eval_name(b->handle, &eval_name);
    if (rc != 0) {
        if (rc != -2) return mk_lgbm_error(env);
        return mk_error(env, "early stopping requires at least one evaluation metric");
    }
    bool maximize = higher_is_better(eval_name);

    int best_iteration = 0;
    int current_iteration = 0;
    double best_score = maximize ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
    int rounds_since_improvement = 0;
    std::vector<double> scores;

    for (int i = 0; i < n; i++) {
        int finished = 0;
        rc = LGBM_BoosterUpdateOneIter(b->handle, &finished);
        if (rc != 0) return mk_lgbm_error(env);

        rc = LGBM_BoosterGetCurrentIteration(b->handle, &current_iteration);
        if (rc != 0) return mk_lgbm_error(env);

        double score = 0.0;
        rc = get_first_valid_eval(b->handle, &score);
        if (rc != 0) {
            if (rc != -2) return mk_lgbm_error(env);
            return mk_error(env, "early stopping requires validation evaluation results");
        }
        scores.push_back(score);

        bool improved = maximize ? score > best_score : score < best_score;
        if (improved) {
            best_score = score;
            best_iteration = current_iteration;
            rounds_since_improvement = 0;
        } else {
            rounds_since_improvement++;
        }

        if (finished || rounds_since_improvement >= stopping_rounds) break;
    }

    double returned_best_score = best_iteration > 0 ? best_score : 0.0;

    if (best_iteration > 0) {
        int rollback_count = current_iteration - best_iteration;
        for (int i = 0; i < rollback_count; i++) {
            rc = LGBM_BoosterRollbackOneIter(b->handle);
            if (rc != 0) return mk_lgbm_error(env);
        }
    }

    ERL_NIF_TERM scores_list = enif_make_list(env, 0);
    for (std::vector<double>::reverse_iterator it = scores.rbegin();
         it != scores.rend(); ++it) {
        scores_list = enif_make_list_cell(env, enif_make_double(env, *it), scores_list);
    }

    ERL_NIF_TERM result = enif_make_tuple4(
        env,
        enif_make_int(env, best_iteration),
        enif_make_double(env, returned_best_score),
        mk_binary(env, eval_name.c_str(), eval_name.size()),
        scores_list);

    return mk_ok(env, result);
}

// booster_predict_for_mat(booster, features_bin, nrow, ncol, params_bin)
//   features_bin: row-major Float64 doubles
//   Returns {:ok, output_binary} where output_binary holds Float64 doubles.
static ERL_NIF_TERM nif_booster_predict_for_mat(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    ErlNifBinary features;
    int nrow, ncol;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[1], &features))
        return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &nrow)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &ncol)) return enif_make_badarg(env);
    char *params = binary_to_cstring(env, argv[4]);
    if (params == NULL) return enif_make_badarg(env);

    if ((size_t)nrow * (size_t)ncol * sizeof(double) != features.size) {
        enif_free(params);
        return mk_error(env, "feature binary size does not match nrow*ncol*8");
    }

    int64_t num_pred = 0;
    int rc = LGBM_BoosterCalcNumPredict(b->handle, nrow, C_API_PREDICT_NORMAL,
                                        /*start_iter=*/0, /*num_iter=*/-1,
                                        &num_pred);
    if (rc != 0) {
        enif_free(params);
        return mk_lgbm_error(env);
    }

    ERL_NIF_TERM out_term;
    unsigned char *out_buf =
        enif_make_new_binary(env, (size_t)num_pred * sizeof(double), &out_term);
    int64_t out_len = 0;
    rc = LGBM_BoosterPredictForMat(b->handle, features.data, C_API_DTYPE_FLOAT64,
                                   nrow, ncol, /*is_row_major=*/1,
                                   C_API_PREDICT_NORMAL, /*start_iter=*/0,
                                   /*num_iter=*/-1, params, &out_len,
                                   (double *)out_buf);
    enif_free(params);
    if (rc != 0) return mk_lgbm_error(env);
    if (out_len != num_pred) return mk_error(env, "predict length mismatch");
    return mk_ok(env, out_term);
}

// booster_get_num_feature(booster)
static ERL_NIF_TERM nif_booster_get_num_feature(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);

    int out = 0;
    int rc = LGBM_BoosterGetNumFeature(b->handle, &out);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_ok(env, enif_make_int(env, out));
}

// booster_get_num_classes(booster)
static ERL_NIF_TERM nif_booster_get_num_classes(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);

    int out = 0;
    int rc = LGBM_BoosterGetNumClasses(b->handle, &out);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_ok(env, enif_make_int(env, out));
}

// booster_get_current_iteration(booster)
static ERL_NIF_TERM nif_booster_get_current_iteration(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);

    int out = 0;
    int rc = LGBM_BoosterGetCurrentIteration(b->handle, &out);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_ok(env, enif_make_int(env, out));
}

// booster_feature_importance(booster, iteration, importance_type)
//   importance_type: C_API_FEATURE_IMPORTANCE_SPLIT or C_API_FEATURE_IMPORTANCE_GAIN
//   Returns {:ok, output_binary} where output_binary holds Float64 doubles.
static ERL_NIF_TERM nif_booster_feature_importance(ErlNifEnv *env, int argc,
                                                   const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    int iteration, importance_type;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &iteration)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &importance_type)) return enif_make_badarg(env);

    int num_features = 0;
    int rc = LGBM_BoosterGetNumFeature(b->handle, &num_features);
    if (rc != 0) return mk_lgbm_error(env);

    ERL_NIF_TERM out_term;
    unsigned char *out_buf = enif_make_new_binary(
        env, (size_t)num_features * sizeof(double), &out_term);

    rc = LGBM_BoosterFeatureImportance(
        b->handle, iteration, importance_type, (double *)out_buf);
    if (rc != 0) return mk_lgbm_error(env);
    return mk_ok(env, out_term);
}

// booster_save_model_to_string(booster, start_iter, num_iter, importance_type)
static ERL_NIF_TERM nif_booster_save_model_to_string(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    HinokiBooster *b;
    int start_iter, num_iter, importance_type;
    if (!enif_get_resource(env, argv[0], HINOKI_BOOSTER_RES, (void **)&b))
        return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &start_iter)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &num_iter)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &importance_type))
        return enif_make_badarg(env);

    int64_t buf_len = 1 << 20; // 1 MiB initial guess
    char *buf = (char *)enif_alloc((size_t)buf_len);
    if (buf == NULL) return mk_error(env, "alloc failed");
    int64_t out_len = 0;
    int rc = LGBM_BoosterSaveModelToString(b->handle, start_iter, num_iter,
                                           importance_type, buf_len, &out_len,
                                           buf);
    if (rc != 0) {
        enif_free(buf);
        return mk_lgbm_error(env);
    }
    if (out_len > buf_len) {
        // Buffer was too small — grow and retry once.
        enif_free(buf);
        buf_len = out_len;
        buf = (char *)enif_alloc((size_t)buf_len);
        if (buf == NULL) return mk_error(env, "alloc failed");
        rc = LGBM_BoosterSaveModelToString(b->handle, start_iter, num_iter,
                                           importance_type, buf_len, &out_len,
                                           buf);
        if (rc != 0) {
            enif_free(buf);
            return mk_lgbm_error(env);
        }
    }
    // out_len includes the trailing NUL — drop it.
    size_t copy_len = (size_t)out_len;
    if (copy_len > 0 && buf[copy_len - 1] == '\0') copy_len--;
    ERL_NIF_TERM result = mk_binary(env, buf, copy_len);
    enif_free(buf);
    return mk_ok(env, result);
}

// booster_load_model_from_string(model_bin)
static ERL_NIF_TERM nif_booster_load_model_from_string(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    char *model = binary_to_cstring(env, argv[0]);
    if (model == NULL) return enif_make_badarg(env);

    int num_iter = 0;
    BoosterHandle out = NULL;
    int rc = LGBM_BoosterLoadModelFromString(model, &num_iter, &out);
    enif_free(model);
    if (rc != 0) return mk_lgbm_error(env);

    HinokiBooster *res = (HinokiBooster *)enif_alloc_resource(
        HINOKI_BOOSTER_RES, sizeof(HinokiBooster));
    res->handle = out;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return mk_ok(env, term);
}

// ---------- table ----------

static ErlNifFunc nif_funcs[] = {
    {"lgbm_version", 0, nif_lgbm_version, 0},
    {"dataset_create_from_mat", 4, nif_dataset_create_from_mat,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"dataset_create_from_mat_reference", 5,
     nif_dataset_create_from_mat_reference, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"dataset_set_label", 2, nif_dataset_set_label,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_create", 2, nif_booster_create, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_add_valid_data", 2, nif_booster_add_valid_data,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_update_iters", 2, nif_booster_update_iters,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_update_iters_early_stopping", 3,
     nif_booster_update_iters_early_stopping, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_predict_for_mat", 5, nif_booster_predict_for_mat,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_get_num_feature", 1, nif_booster_get_num_feature, 0},
    {"booster_get_num_classes", 1, nif_booster_get_num_classes, 0},
    {"booster_get_current_iteration", 1, nif_booster_get_current_iteration, 0},
    {"booster_feature_importance", 3, nif_booster_feature_importance,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_save_model_to_string", 4, nif_booster_save_model_to_string,
     ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"booster_load_model_from_string", 1, nif_booster_load_model_from_string,
     ERL_NIF_DIRTY_JOB_IO_BOUND}};

ERL_NIF_INIT(Elixir.Hinoki.NIF, nif_funcs, load, NULL, upgrade, NULL)
