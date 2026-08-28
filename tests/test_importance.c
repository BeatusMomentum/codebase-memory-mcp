/*
 * test_importance.c — index-time per-symbol importance score (pass_importance.c).
 *
 * Formula under test:
 *   importance = sqrt(num_refs) * priv * generic * distinct * test_penalty
 *
 * Three of these tests exist to catch failure modes that are SILENT — they
 * produce a wrong index with a fully green build:
 *
 *   - importance_pass_actually_runs_in_full_pipeline
 *       binds the predump registration. A hand-written PREDUMP_PASS_COUNT that
 *       lags the pass table skips the LAST-registered pass and nothing else
 *       notices. This test fails loudly if the score never lands.
 *   - importance_rehydrated_node_keeps_exactly_one_key
 *   - importance_second_index_run_keeps_exactly_one_key
 *       bind the idempotent write-back. The incremental path rehydrates nodes
 *       that already carry "importance"; a plain append yields
 *       {"importance":1.0,...,"importance":2.0} — property corruption, not a
 *       build failure.
 *   - importance_distinct_file_count_is_linear_in_group_size
 *       binds the cost shape: the distinct-file count must be computed once
 *       per distinct NAME, not once per node. The per-node shape is O(k^3) in
 *       a same-name group and is measurably disqualifying on real corpora.
 */
#include "test_framework.h"
#include "test_helpers.h"

#include "pipeline/pipeline.h"
#include "pipeline/pipeline_internal.h"
#include "graph_buffer/graph_buffer.h"
#include "store/store.h"
#include "foundation/compat.h"
#include "foundation/log.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Shared helpers ───────────────────────────────────────────────────── */

/* Parse the numeric value of "importance" out of a properties_json blob.
 * Returns a sentinel (-999.0) when the key is absent — callers that care about
 * presence assert on imp_key_count() instead. */
static double imp_value(const char *json) {
    if (!json) {
        return -999.0;
    }
    const char *p = strstr(json, "\"importance\":");
    if (!p) {
        return -999.0;
    }
    return strtod(p + strlen("\"importance\":"), NULL);
}

/* How many times the key appears. The duplicate-key corruption this suite
 * guards against shows up here as 2. */
static int imp_key_count(const char *json) {
    if (!json) {
        return 0;
    }
    int n = 0;
    const char *p = json;
    while ((p = strstr(p, "\"importance\":")) != NULL) {
        n++;
        p += strlen("\"importance\":");
    }
    return n;
}

static void imp_run_pass(cbm_gbuf_t *gb) {
    atomic_int cancelled = 0;
    cbm_pipeline_ctx_t ctx = {
        .project_name = "test-proj",
        .repo_path = "/tmp/test",
        .gbuf = gb,
        .cancelled = &cancelled,
    };
    cbm_pipeline_pass_importance(&ctx);
}

/* Every stored node of the given label must carry exactly `expect` copies of
 * the key. Returns the number of nodes checked, or -1 on a store error. */
static int imp_check_label(cbm_store_t *s, const char *project, const char *label, int expect) {
    cbm_node_t *nodes = NULL;
    int count = 0;
    if (cbm_store_find_nodes_by_label(s, project, label, &nodes, &count) != CBM_STORE_OK) {
        return -1;
    }
    int bad = 0;
    for (int i = 0; i < count; i++) {
        if (imp_key_count(nodes[i].properties_json) != expect) {
            bad++;
            printf("    %s node '%s' props=%s\n", label, nodes[i].name ? nodes[i].name : "?",
                   nodes[i].properties_json ? nodes[i].properties_json : "(null)");
        }
    }
    cbm_store_free_nodes(nodes, count);
    return bad == 0 ? count : -1;
}

/* ── Registration: the pass must actually execute ─────────────────────── */

/* A miscounted PREDUMP_PASS_COUNT silently drops the last-registered pass.
 * The score is the only observable, so assert on the score. */
TEST(importance_pass_actually_runs_in_full_pipeline) {
    char *tmp = th_mktempdir("cbm_imp_run");
    if (!tmp) {
        FAIL("tmpdir");
    }
    char root[512];
    snprintf(root, sizeof(root), "%s", tmp); /* th_mktempdir returns a static buffer */

    ASSERT_EQ(th_write_file(TH_PATH(root, "main.py"),
                            "def helper():\n    return 1\n\n"
                            "def caller():\n    return helper() + helper()\n\n"
                            "class Widget:\n    def render(self):\n        return helper()\n"),
              0);

    char db_path[600];
    snprintf(db_path, sizeof(db_path), "%s/graph.db", root);
    cbm_pipeline_t *p = cbm_pipeline_new(root, db_path, CBM_MODE_FULL);
    ASSERT_NOT_NULL(p);
    ASSERT_EQ(cbm_pipeline_run(p), 0);
    char project[256];
    snprintf(project, sizeof(project), "%s", cbm_pipeline_project_name(p));
    cbm_pipeline_free(p);

    cbm_store_t *s = cbm_store_open_path(db_path);
    ASSERT_NOT_NULL(s);

    /* Non-vacuous: the corpus really produced symbols of each kind. */
    int fc = imp_check_label(s, project, "Function", 1);
    int mc = imp_check_label(s, project, "Method", 1);
    int cc = imp_check_label(s, project, "Class", 1);
    printf("    scored Function=%d Method=%d Class=%d\n", fc, mc, cc);
    ASSERT_GT(fc, 0);
    ASSERT_GT(mc, 0);
    ASSERT_GT(cc, 0);

    /* File nodes are deliberately excluded from scoring. */
    cbm_node_t *files = NULL;
    int filec = 0;
    ASSERT_EQ(cbm_store_find_nodes_by_label(s, project, "File", &files, &filec), CBM_STORE_OK);
    for (int i = 0; i < filec; i++) {
        ASSERT_EQ(imp_key_count(files[i].properties_json), 0);
    }
    cbm_store_free_nodes(files, filec);

    cbm_store_close(s);
    th_rmtree(root);
    PASS();
}

/* ── Formula ──────────────────────────────────────────────────────────── */

TEST(importance_base_is_sqrt_of_incoming_refs) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);

    int64_t target =
        cbm_gbuf_upsert_node(gb, "Function", "target", "pkg.target", "pkg/main.go", 1, 1, "{}");
    int64_t c1 = cbm_gbuf_upsert_node(gb, "Function", "c1", "pkg.c1", "pkg/main.go", 2, 2, "{}");
    int64_t c2 = cbm_gbuf_upsert_node(gb, "Function", "c2", "pkg.c2", "pkg/main.go", 3, 3, "{}");
    int64_t c3 = cbm_gbuf_upsert_node(gb, "Function", "c3", "pkg.c3", "pkg/main.go", 4, 4, "{}");
    int64_t lonely =
        cbm_gbuf_upsert_node(gb, "Function", "lonely", "pkg.lonely", "pkg/main.go", 5, 5, "{}");
    ASSERT_GT(target, 0);
    ASSERT_GT(lonely, 0);

    cbm_gbuf_insert_edge(gb, c1, target, "CALLS", "{}");
    cbm_gbuf_insert_edge(gb, c2, target, "CALLS", "{}");
    cbm_gbuf_insert_edge(gb, c3, target, "USAGE", "{}"); /* spans BOTH edge types */

    imp_run_pass(gb);

    const cbm_gbuf_node_t *tn = cbm_gbuf_find_by_id(gb, target);
    ASSERT_NOT_NULL(tn);
    ASSERT_EQ(imp_key_count(tn->properties_json), 1);
    ASSERT_FLOAT_EQ(imp_value(tn->properties_json), sqrt(3.0), 1e-6);

    /* num_refs == 0 -> sqrt(0) == 0, still written. */
    const cbm_gbuf_node_t *ln = cbm_gbuf_find_by_id(gb, lonely);
    ASSERT_NOT_NULL(ln);
    ASSERT_EQ(imp_key_count(ln->properties_json), 1);
    ASSERT_FLOAT_EQ(imp_value(ln->properties_json), 0.0, 1e-9);

    cbm_gbuf_free(gb);
    PASS();
}

TEST(importance_private_names_are_demoted) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);

    int64_t priv =
        cbm_gbuf_upsert_node(gb, "Function", "_run", "pkg._run", "pkg/main.go", 1, 1, "{}");
    int64_t pub = cbm_gbuf_upsert_node(gb, "Function", "run", "pkg.run", "pkg/main.go", 2, 2, "{}");
    for (int i = 0; i < 4; i++) {
        char name[CBM_SZ_32];
        char qn[CBM_SZ_64];
        snprintf(name, sizeof(name), "p%d", i);
        snprintf(qn, sizeof(qn), "pkg.p%d", i);
        int64_t c =
            cbm_gbuf_upsert_node(gb, "Function", name, qn, "pkg/main.go", 10 + i, 10 + i, "{}");
        cbm_gbuf_insert_edge(gb, c, priv, "CALLS", "{}");
        snprintf(name, sizeof(name), "q%d", i);
        snprintf(qn, sizeof(qn), "pkg.q%d", i);
        c = cbm_gbuf_upsert_node(gb, "Function", name, qn, "pkg/main.go", 20 + i, 20 + i, "{}");
        cbm_gbuf_insert_edge(gb, c, pub, "CALLS", "{}");
    }

    imp_run_pass(gb);

    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, priv)->properties_json), sqrt(4.0) * 0.1,
                    1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, pub)->properties_json), sqrt(4.0), 1e-6);

    cbm_gbuf_free(gb);
    PASS();
}

/* Generic = defined in >= 5 DISTINCT FILES. Two defs of one name in the same
 * file count once, so the boundary is tested on files, not on node count. */
TEST(importance_generic_names_are_demoted_on_distinct_files) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);

    /* "generic": 5 distinct files -> demoted. */
    int64_t generic = 0;
    for (int i = 0; i < 5; i++) {
        char qn[CBM_SZ_64];
        char file[CBM_SZ_64];
        snprintf(qn, sizeof(qn), "pkg%d.generic", i);
        snprintf(file, sizeof(file), "pkg%d/main.go", i);
        int64_t id = cbm_gbuf_upsert_node(gb, "Function", "generic", qn, file, 1, 1, "{}");
        if (i == 0) {
            generic = id;
        }
    }
    /* "narrow": 4 distinct files, but 6 nodes — two duplicates share a file,
     * so the DISTINCT count is 4 and it stays undemoted. */
    int64_t narrow = 0;
    for (int i = 0; i < 4; i++) {
        char qn[CBM_SZ_64];
        char file[CBM_SZ_64];
        snprintf(qn, sizeof(qn), "n%d.narrow", i);
        snprintf(file, sizeof(file), "n%d/main.go", i);
        int64_t id = cbm_gbuf_upsert_node(gb, "Function", "narrow", qn, file, 1, 1, "{}");
        if (i == 0) {
            narrow = id;
        }
    }
    cbm_gbuf_upsert_node(gb, "Function", "narrow", "n0.narrow_b", "n0/main.go", 9, 9, "{}");
    cbm_gbuf_upsert_node(gb, "Function", "narrow", "n1.narrow_b", "n1/main.go", 9, 9, "{}");

    for (int i = 0; i < 9; i++) {
        char name[CBM_SZ_32];
        char qn[CBM_SZ_64];
        snprintf(name, sizeof(name), "z%d", i);
        snprintf(qn, sizeof(qn), "z.z%d", i);
        int64_t c = cbm_gbuf_upsert_node(gb, "Function", name, qn, "z/main.go", i, i, "{}");
        cbm_gbuf_insert_edge(gb, c, i < 4 ? generic : narrow, "CALLS", "{}");
    }

    imp_run_pass(gb);

    /* generic: sqrt(4) * 0.1;  narrow: sqrt(5), undemoted. */
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, generic)->properties_json), sqrt(4.0) * 0.1,
                    1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, narrow)->properties_json), sqrt(5.0), 1e-6);

    cbm_gbuf_free(gb);
    PASS();
}

TEST(importance_distinctive_names_are_promoted) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);

    /* snake_case >= 8, camelCase >= 8, and a plain long lowercase word that is
     * NOT distinctive (no '_', no hump). */
    int64_t snake = cbm_gbuf_upsert_node(gb, "Function", "load_user_profile",
                                         "pkg.load_user_profile", "pkg/a.go", 1, 1, "{}");
    int64_t camel = cbm_gbuf_upsert_node(gb, "Function", "loadUserProfile", "pkg.loadUserProfile",
                                         "pkg/b.go", 1, 1, "{}");
    int64_t plain =
        cbm_gbuf_upsert_node(gb, "Function", "aggregate", "pkg.aggregate", "pkg/c.go", 1, 1, "{}");
    int64_t shortnm =
        cbm_gbuf_upsert_node(gb, "Function", "get_x", "pkg.get_x", "pkg/d.go", 1, 1, "{}");

    const int64_t targets[] = {snake, camel, plain, shortnm};
    for (int t = 0; t < 4; t++) {
        char name[CBM_SZ_32];
        char qn[CBM_SZ_64];
        snprintf(name, sizeof(name), "cc%d", t);
        snprintf(qn, sizeof(qn), "z.cc%d", t);
        int64_t c = cbm_gbuf_upsert_node(gb, "Function", name, qn, "z/main.go", t, t, "{}");
        cbm_gbuf_insert_edge(gb, c, targets[t], "CALLS", "{}");
    }

    imp_run_pass(gb);

    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, snake)->properties_json), 1.0 * 10.0, 1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, camel)->properties_json), 1.0 * 10.0, 1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, plain)->properties_json), 1.0, 1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, shortnm)->properties_json), 1.0, 1e-6);

    cbm_gbuf_free(gb);
    PASS();
}

/* Test scaffolding is demoted two ways: living in a test file (the graph's own
 * cbm_is_test_path classifier) or being the target of a TESTS edge. */
TEST(importance_test_scaffolding_is_demoted) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);

    int64_t in_test_file = cbm_gbuf_upsert_node(gb, "Function", "fixture", "t.fixture",
                                                "pkg/thing_test.go", 1, 1, "{}");
    int64_t tests_target =
        cbm_gbuf_upsert_node(gb, "Function", "helper", "pkg.helper", "pkg/util.go", 1, 1, "{}");
    int64_t plain =
        cbm_gbuf_upsert_node(gb, "Function", "worker", "pkg.worker", "pkg/util.go", 2, 2, "{}");
    /* A production file whose NAME merely contains "test" must NOT be demoted. */
    int64_t testutil = cbm_gbuf_upsert_node(gb, "Function", "spawn", "pkg.spawn",
                                            "pkg/testutil_helpers.go", 1, 1, "{}");

    const int64_t targets[] = {in_test_file, tests_target, plain, testutil};
    for (int t = 0; t < 4; t++) {
        char name[CBM_SZ_32];
        char qn[CBM_SZ_64];
        snprintf(name, sizeof(name), "d%d", t);
        snprintf(qn, sizeof(qn), "z.d%d", t);
        int64_t c = cbm_gbuf_upsert_node(gb, "Function", name, qn, "z/main.go", t, t, "{}");
        cbm_gbuf_insert_edge(gb, c, targets[t], "CALLS", "{}");
    }
    int64_t tfn = cbm_gbuf_upsert_node(gb, "Function", "TestHelper", "t.TestHelper",
                                       "pkg/util_test.go", 5, 5, "{}");
    cbm_gbuf_insert_edge(gb, tfn, tests_target, "TESTS", "{}");

    imp_run_pass(gb);

    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, in_test_file)->properties_json), 1.0 * 0.1,
                    1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, tests_target)->properties_json), 1.0 * 0.1,
                    1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, plain)->properties_json), 1.0, 1e-6);
    ASSERT_FLOAT_EQ(imp_value(cbm_gbuf_find_by_id(gb, testutil)->properties_json), 1.0, 1e-6);

    cbm_gbuf_free(gb);
    PASS();
}

/* ── Idempotent write-back ────────────────────────────────────────────── */

TEST(importance_append_prop_overwrites_instead_of_duplicating) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);
    int64_t id = cbm_gbuf_upsert_node(gb, "Function", "f", "p.f", "p/a.go", 1, 1,
                                      "{\"loop_depth\":2,\"recursive\":false}");
    cbm_gbuf_node_t *n = (cbm_gbuf_node_t *)cbm_gbuf_find_by_id(gb, id);
    ASSERT_NOT_NULL(n);

    cbm_pipeline_importance_append_prop(n, 1.5);
    ASSERT_EQ(imp_key_count(n->properties_json), 1);
    ASSERT_FLOAT_EQ(imp_value(n->properties_json), 1.5, 1e-6);

    cbm_pipeline_importance_append_prop(n, 42.25);
    printf("    after second write: %s\n", n->properties_json);
    ASSERT_EQ(imp_key_count(n->properties_json), 1);
    ASSERT_FLOAT_EQ(imp_value(n->properties_json), 42.25, 1e-6);

    /* Sibling keys survive the in-place overwrite, on both sides of the key. */
    ASSERT_TRUE(strstr(n->properties_json, "\"loop_depth\":2") != NULL);
    ASSERT_TRUE(strstr(n->properties_json, "\"recursive\":false") != NULL);

    /* A key that is NOT last: overwrite must not truncate the tail. */
    free(n->properties_json);
    n->properties_json = strdup("{\"importance\":0.000000,\"tail\":7}");
    ASSERT_NOT_NULL(n->properties_json);
    cbm_pipeline_importance_append_prop(n, 3.0);
    ASSERT_EQ(imp_key_count(n->properties_json), 1);
    ASSERT_FLOAT_EQ(imp_value(n->properties_json), 3.0, 1e-6);
    ASSERT_TRUE(strstr(n->properties_json, "\"tail\":7") != NULL);

    /* A string VALUE that merely contains the key text is not a key. */
    free(n->properties_json);
    n->properties_json = strdup("{\"doc\":\"see \\\"importance\\\": below\"}");
    ASSERT_NOT_NULL(n->properties_json);
    cbm_pipeline_importance_append_prop(n, 2.0);
    ASSERT_FLOAT_EQ(imp_value(n->properties_json), 2.0, 1e-6);
    ASSERT_TRUE(strstr(n->properties_json, "\"doc\":") != NULL);

    /* Empty object and a non-object blob. */
    free(n->properties_json);
    n->properties_json = strdup("{}");
    ASSERT_NOT_NULL(n->properties_json);
    cbm_pipeline_importance_append_prop(n, 0.5);
    ASSERT_STR_EQ(n->properties_json, "{\"importance\":0.500000}");

    free(n->properties_json);
    n->properties_json = strdup("not json");
    ASSERT_NOT_NULL(n->properties_json);
    cbm_pipeline_importance_append_prop(n, 9.0);
    ASSERT_STR_EQ(n->properties_json, "not json");

    cbm_gbuf_free(gb);
    PASS();
}

/* Running the pass twice over the same graph — the shape the incremental path
 * produces when it re-scores an already-scored rehydrated buffer. */
TEST(importance_pass_is_idempotent_across_reruns) {
    cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
    ASSERT_NOT_NULL(gb);
    int64_t t =
        cbm_gbuf_upsert_node(gb, "Function", "target", "pkg.target", "pkg/main.go", 1, 1, "{}");
    int64_t c =
        cbm_gbuf_upsert_node(gb, "Function", "caller", "pkg.caller", "pkg/main.go", 2, 2, "{}");
    cbm_gbuf_insert_edge(gb, c, t, "CALLS", "{}");

    imp_run_pass(gb);
    char first[CBM_SZ_512];
    snprintf(first, sizeof(first), "%s", cbm_gbuf_find_by_id(gb, t)->properties_json);
    imp_run_pass(gb);
    const char *second = cbm_gbuf_find_by_id(gb, t)->properties_json;

    printf("    run1=%s run2=%s\n", first, second);
    ASSERT_EQ(imp_key_count(second), 1);
    ASSERT_STR_EQ(first, second); /* re-scoring is a fixed point */

    cbm_gbuf_free(gb);
    PASS();
}

/* The literal incremental shape: index, rehydrate the stored graph exactly as
 * pipeline_incremental.c does, re-run the pass. Nodes arrive already carrying
 * the key — an append-only write-back yields two. */
TEST(importance_rehydrated_node_keeps_exactly_one_key) {
    char *tmp = th_mktempdir("cbm_imp_rehy");
    if (!tmp) {
        FAIL("tmpdir");
    }
    char root[512];
    snprintf(root, sizeof(root), "%s", tmp); /* th_mktempdir returns a static buffer */

    ASSERT_EQ(th_write_file(TH_PATH(root, "main.py"),
                            "def helper():\n    return 1\n\n"
                            "def caller():\n    return helper() + helper()\n"),
              0);
    char db_path[600];
    snprintf(db_path, sizeof(db_path), "%s/graph.db", root);
    cbm_pipeline_t *p = cbm_pipeline_new(root, db_path, CBM_MODE_FULL);
    ASSERT_NOT_NULL(p);
    ASSERT_EQ(cbm_pipeline_run(p), 0);
    char project[256];
    snprintf(project, sizeof(project), "%s", cbm_pipeline_project_name(p));
    cbm_pipeline_free(p);

    cbm_gbuf_t *gb = cbm_gbuf_new(project, root);
    ASSERT_NOT_NULL(gb);
    ASSERT_EQ(cbm_gbuf_load_from_db(gb, db_path, project), 0);

    /* Precondition: the rehydrated nodes already carry the key. Without this
     * the test could pass while proving nothing. */
    const cbm_gbuf_node_t **fns = NULL;
    int fc = 0;
    ASSERT_EQ(cbm_gbuf_find_by_label(gb, "Function", &fns, &fc), 0);
    ASSERT_GT(fc, 0);
    for (int i = 0; i < fc; i++) {
        ASSERT_EQ(imp_key_count(fns[i]->properties_json), 1);
    }

    imp_run_pass(gb);

    ASSERT_EQ(cbm_gbuf_find_by_label(gb, "Function", &fns, &fc), 0);
    ASSERT_GT(fc, 0);
    for (int i = 0; i < fc; i++) {
        if (imp_key_count(fns[i]->properties_json) != 1) {
            printf("    duplicate key on '%s': %s\n", fns[i]->name, fns[i]->properties_json);
        }
        ASSERT_EQ(imp_key_count(fns[i]->properties_json), 1);
    }

    cbm_gbuf_free(gb);
    th_rmtree(root);
    PASS();
}

/* Pass-timing observer: which importance-pass call sites fired. */
static atomic_int g_saw_incr_importance = 0;
static atomic_int g_saw_full_importance = 0;
static void imp_pass_sink(const char *line) {
    if (!line || !strstr(line, "pass.timing")) {
        return;
    }
    if (strstr(line, "pass=incr_importance")) {
        atomic_fetch_add_explicit(&g_saw_incr_importance, 1, memory_order_relaxed);
    } else if (strstr(line, "pass=importance")) {
        atomic_fetch_add_explicit(&g_saw_full_importance, 1, memory_order_relaxed);
    }
}

/* End-to-end, on the incremental route that actually rehydrates the stored
 * graph (LEGACY_PARTIAL — production's other incremental route, closure
 * repair, works on a delta buffer and never sees pre-scored nodes). Index,
 * edit a file, index again: the run must be observed on that route, the pass
 * must have run on it, and every stored symbol must carry exactly one key. An
 * append-only write-back produces two here. */
TEST(importance_second_index_run_keeps_exactly_one_key) {
    char *tmp = th_mktempdir("cbm_imp_incr");
    if (!tmp) {
        FAIL("tmpdir");
    }
    char root[512];
    snprintf(root, sizeof(root), "%s", tmp); /* th_mktempdir returns a static buffer */

    ASSERT_EQ(th_write_file(TH_PATH(root, "a.py"), "def helper():\n    return 1\n"), 0);
    ASSERT_EQ(
        th_write_file(TH_PATH(root, "b.py"), "import a\n\ndef caller():\n    return a.helper()\n"),
        0);

    char db_path[600];
    snprintf(db_path, sizeof(db_path), "%s/graph.db", root);

    atomic_store(&g_saw_incr_importance, 0);
    atomic_store(&g_saw_full_importance, 0);
    cbm_log_set_sink_ex(imp_pass_sink, CBM_LOG_SINK_TEE);

    cbm_pipeline_t *p1 = cbm_pipeline_new(root, db_path, CBM_MODE_FULL);
    ASSERT_NOT_NULL(p1);
    ASSERT_EQ(cbm_pipeline_run(p1), 0);
    char project[256];
    snprintf(project, sizeof(project), "%s", cbm_pipeline_project_name(p1));
    cbm_pipeline_free(p1);

    /* Edit one file so the second run has real work but stays incremental. */
    ASSERT_EQ(th_write_file(TH_PATH(root, "b.py"),
                            "import a\n\ndef caller():\n    return a.helper() + a.helper()\n"),
              0);

    cbm_pipeline_incremental_test_force_legacy_partial_once();
    cbm_pipeline_t *p2 = cbm_pipeline_new(root, db_path, CBM_MODE_FULL);
    ASSERT_NOT_NULL(p2);
    ASSERT_EQ(cbm_pipeline_run(p2), 0);
    cbm_incremental_route_t route = cbm_pipeline_incremental_test_last_route();
    cbm_pipeline_free(p2);
    cbm_log_set_sink(NULL);

    /* Route proof: without it a silent fall-back to a full reindex would make
     * the duplicate-key assertion below vacuous. */
    ASSERT_EQ((int)route, (int)CBM_INCREMENTAL_ROUTE_LEGACY_PARTIAL);

    int incr = atomic_load(&g_saw_incr_importance);
    int full = atomic_load(&g_saw_full_importance);
    printf("    importance pass runs: full=%d incremental=%d\n", full, incr);
    ASSERT_GT(full, 0); /* first run scored via the predump table */
    /* The second run must have re-scored too — through whichever path it took.
     * If it was NOT incremental the duplicate-key trap is untested, so say so
     * rather than passing quietly. */
    if (incr == 0) {
        th_rmtree(root);
        FAIL("second run did not re-score on the incremental path — trap untested");
    }

    cbm_store_t *s = cbm_store_open_path(db_path);
    ASSERT_NOT_NULL(s);
    int fc = imp_check_label(s, project, "Function", 1);
    printf("    functions with exactly one importance key: %d\n", fc);
    ASSERT_GT(fc, 0);
    cbm_store_close(s);

    th_rmtree(root);
    PASS();
}

/* ── Cost shape ───────────────────────────────────────────────────────── */

/* The distinct-file count must be computed once per distinct NAME. Computing
 * it per NODE makes a same-name group of size k cost O(k^3) — measurably
 * disqualifying on real corpora. Assert on the pass's own work counter, not
 * on wall time: doubling a same-name group must roughly double the work. */
TEST(importance_distinct_file_count_is_linear_in_group_size) {
    enum { K = 200 };

    uint64_t visits[2];
    for (int leg = 0; leg < 2; leg++) {
        int k = (leg == 0) ? K : 2 * K;
        cbm_gbuf_t *gb = cbm_gbuf_new("test-proj", "/tmp/test");
        ASSERT_NOT_NULL(gb);
        for (int i = 0; i < k; i++) {
            char qn[CBM_SZ_64];
            char file[CBM_SZ_64];
            snprintf(qn, sizeof(qn), "pkg%d.toString", i);
            snprintf(file, sizeof(file), "pkg%d/a.java", i);
            ASSERT_GT(cbm_gbuf_upsert_node(gb, "Method", "toString", qn, file, 1, 1, "{}"), 0);
        }
        uint64_t v0 = atomic_load_explicit(&g_importance_name_visits, memory_order_relaxed);
        imp_run_pass(gb);
        visits[leg] = atomic_load_explicit(&g_importance_name_visits, memory_order_relaxed) - v0;
        cbm_gbuf_free(gb);
    }

    double ratio = (visits[0] > 0) ? (double)visits[1] / (double)visits[0] : 0.0;
    printf("    name visits k=%d -> %llu, k=%d -> %llu (ratio %.2f; linear ~2, per-node ~4)\n", K,
           (unsigned long long)visits[0], 2 * K, (unsigned long long)visits[1], ratio);

    /* Non-vacuous: the group really was scanned. */
    ASSERT_GT((long long)visits[0], 0);
    /* Memoized: one scan of the whole group, so visits == k exactly. */
    ASSERT_EQ((long long)visits[0], (long long)K);
    ASSERT_EQ((long long)visits[1], (long long)(2 * K));
    ASSERT_TRUE(ratio <= 2.75);
    PASS();
}

/* ── Suite ────────────────────────────────────────────────────────────── */

SUITE(importance) {
    /* Registration — catches a miscounted PREDUMP_PASS_COUNT. */
    RUN_TEST(importance_pass_actually_runs_in_full_pipeline);

    /* Formula. */
    RUN_TEST(importance_base_is_sqrt_of_incoming_refs);
    RUN_TEST(importance_private_names_are_demoted);
    RUN_TEST(importance_generic_names_are_demoted_on_distinct_files);
    RUN_TEST(importance_distinctive_names_are_promoted);
    RUN_TEST(importance_test_scaffolding_is_demoted);

    /* Idempotent write-back — the incremental duplicate-key trap. */
    RUN_TEST(importance_append_prop_overwrites_instead_of_duplicating);
    RUN_TEST(importance_pass_is_idempotent_across_reruns);
    RUN_TEST(importance_rehydrated_node_keeps_exactly_one_key);
    RUN_TEST(importance_second_index_run_keeps_exactly_one_key);

    /* Cost shape. */
    RUN_TEST(importance_distinct_file_count_is_linear_in_group_size);
}
