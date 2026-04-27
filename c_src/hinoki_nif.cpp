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

// dataset_create_from_mat(features_bin, nrow, ncol, params_bin)
//   features_bin: row-major Float64 doubles, length = nrow*ncol*8 bytes
static ERL_NIF_TERM nif_dataset_create_from_mat(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary features;
    int nrow, ncol;
    if (!enif_inspect_binary(env, argv[0], &features)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &nrow)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &ncol)) return enif_make_badarg(env);

    char *params = binary_to_cstring(env, argv[3]);
    if (params == NULL) return enif_make_badarg(env);

    if ((size_t)nrow * (size_t)ncol * sizeof(double) != features.size) {
        enif_free(params);
        return mk_error(env, "feature binary size does not match nrow*ncol*8");
    }

    DatasetHandle out = NULL;
    int rc = LGBM_DatasetCreateFromMat(features.data, C_API_DTYPE_FLOAT64,
                                       nrow, ncol, /*is_row_major=*/1,
                                       params, /*reference=*/NULL, &out);
    enif_free(params);
    if (rc != 0) return mk_lgbm_error(env);

    HinokiDataset *res = (HinokiDataset *)enif_alloc_resource(
        HINOKI_DATASET_RES, sizeof(HinokiDataset));
    res->handle = out;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return mk_ok(env, term);
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
    {"dataset_set_label", 2, nif_dataset_set_label,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_create", 2, nif_booster_create, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_update_iters", 2, nif_booster_update_iters,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_predict_for_mat", 5, nif_booster_predict_for_mat,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"booster_save_model_to_string", 4, nif_booster_save_model_to_string,
     ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"booster_load_model_from_string", 1, nif_booster_load_model_from_string,
     ERL_NIF_DIRTY_JOB_IO_BOUND}};

ERL_NIF_INIT(Elixir.Hinoki.NIF, nif_funcs, load, NULL, upgrade, NULL)
