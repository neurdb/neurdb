-- script to define tables and functions for nr_model

-- create tables
CREATE TABLE model
(
    model_id   serial PRIMARY KEY,
    model_meta bytea NOT NULL
);

CREATE TABLE layer
(
    model_id int REFERENCES model(model_id),
    layer_id int,
    create_time timestamp,
    layer_data bytea
);

-- save and load functions
-- @see {src/access/model_sl.h}
-- CREATE FUNCTION pgm_register_model(model_name text, model_path text) RETURNS BOOL
-- AS
-- 'MODULE_PATHNAME',
-- 'pgm_register_model'
--     LANGUAGE C STRICT;

-- @see {src/access/model_sl.h}
-- CREATE FUNCTION pgm_unregister_model(model_name text) RETURNS BOOL
-- AS
-- 'MODULE_PATHNAME',
-- 'pgm_unregister_model'
--     LANGUAGE C STRICT;

-- @see {src/access/model_sl.h}
-- CREATE FUNCTION pgm_store_model(model_name text, model_path text) RETURNS BOOL
-- AS
-- 'MODULE_PATHNAME',
-- 'pgm_store_model'
--     LANGUAGE C STRICT;

-- @see {src/access/model_sl.h}
-- CREATE FUNCTION pgm_get_model_id_by_name(model_name text) RETURNS INT
-- AS
-- 'MODULE_PATHNAME',
-- 'pgm_get_model_id_by_name'
--     LANGUAGE C STRICT;

-- inference functions
-- @see {src/inference/model_inference.h}
-- the input is an array of float4
CREATE FUNCTION pgm_predict_float4(model_name text, input anyarray)
    RETURNS SETOF RECORD
AS
'MODULE_PATHNAME',
'pgm_predict_float4'
    LANGUAGE C STRICT;

-- test function
CREATE FUNCTION pgm_predict_table(model_name text, batch_size int, table_name text, column_names text[])
    RETURNS SETOF RECORD
AS
'MODULE_PATHNAME',
'pgm_predict_table'
    LANGUAGE C STRICT;
