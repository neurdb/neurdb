CREATE FUNCTION nr_pipeline_init (
    model_name text,
    table_name text,
    batch_size int,
    epoch int,
    nfeat int,
    feature_names text[],
    n_features int,
    target text,
    type int,
    tupdesc anyelement
) RETURNS VOID AS 'MODULE_PATHNAME',
'nr_pipeline_init' LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION nr_pipeline_push_slot (slot anyelement, flush boolean) RETURNS text[] AS 'MODULE_PATHNAME',
'nr_pipeline_push_slot' LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION nr_pipeline_state_change (to_inference boolean) RETURNS VOID AS 'MODULE_PATHNAME',
'nr_pipeline_state_change' LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION nr_pipeline_close () RETURNS VOID AS 'MODULE_PATHNAME',
'nr_pipeline_close' LANGUAGE C STRICT VOLATILE;
