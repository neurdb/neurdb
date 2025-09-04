/* define tables and functions for pg_neurstore PostgreSQL extension */
CREATE TABLE model
(
    model_id   serial PRIMARY KEY,
    model_name text  NOT NULL
);

/* create functions */
/* save model function */
CREATE FUNCTION ns_save_model(
    model_name text,
    tolerance float4,
    model_path text
) RETURNS BOOL
AS
'MODULE_PATHNAME',
'ns_save_model'
    LANGUAGE C STRICT
               STABLE;

/* save models */
CREATE FUNCTION ns_save_models(
    model_names text[],
    tolerance float4,
    model_path text
) RETURNS BOOL
AS
'MODULE_PATHNAME',
'ns_save_models'
    LANGUAGE C STRICT
               STABLE;

/* save models (.onnx) from a folder */
CREATE FUNCTION ns_save_models_from_folder(
    folder_path text,
    tolerance float4
) RETURNS BOOL
AS
'MODULE_PATHNAME',
'ns_save_models_from_folder'
    LANGUAGE C STRICT
               STABLE;

/* load model */
CREATE FUNCTION ns_load_model(
    model_id int,
    return_serialized BOOL DEFAULT FALSE
) RETURNS BYTEA
AS
'MODULE_PATHNAME',
'ns_load_model'
    LANGUAGE C STRICT
               STABLE;

/* flexible loading - base-8bit */
CREATE FUNCTION ns_load_model_as_uint8(
    model_id int,
    return_serialized BOOL DEFAULT FALSE
) RETURNS BYTEA
AS
'MODULE_PATHNAME',
'ns_load_model_as_uint8'
    LANGUAGE C STRICT
               STABLE;

/* flexible loading - base-8bit-delta */
CREATE FUNCTION ns_load_model_as_uint8_delta(
    model_id int,
    return_serialized BOOL DEFAULT FALSE
) RETURNS BYTEA
AS
'MODULE_PATHNAME',
'ns_load_model_as_uint8_delta'
    LANGUAGE C STRICT
               STABLE;

/* float16 */
CREATE FUNCTION ns_load_model_as_float16(
    model_id int,
    return_serialized BOOL DEFAULT FALSE
) RETURNS BYTEA
AS
'MODULE_PATHNAME',
'ns_load_model_as_float16'
    LANGUAGE C STRICT
               STABLE;

CREATE OR REPLACE FUNCTION ns_inference(
    model_id          int4,
    query             text,
    pg_dsn            text,
    input_columns     text[],
    output_column     text,
    batch_size        int4,
    tokenizer_path    text,
    pad_token         text,
    eos_token         text,
    bos_token         text,
    max_input_len     int4,
    max_output_len    int4,
    load_mode         int4  DEFAULT 0,     -- 0:Float32 1:Float16 2:Int8 3:Int8Delta
    task              int4  DEFAULT 0,     -- 0:SequenceClassification
    use_gpu           bool DEFAULT false
) RETURNS bool
AS
'MODULE_PATHNAME',
'ns_inference'
LANGUAGE C IMMUTABLE PARALLEL SAFE STRICT;

CREATE OR REPLACE FUNCTION ns_save_model_dry_run(
    tolerance        real,
    model_path       text,
    load_mode        int4,
    query            text,
    pg_dsn           text,
    input_columns    text[],
    output_column    text,
    batch_size       int4,
    tokenizer_path   text,
    pad_token        text,
    eos_token        text,
    bos_token        text,
    max_input_len    int4,
    max_output_len   int4,
    task             int4,
    use_gpu          boolean
) RETURNS float8[]
AS 'MODULE_PATHNAME', 'ns_save_model_dry_run'
LANGUAGE C IMMUTABLE PARALLEL SAFE STRICT;

CREATE FUNCTION ns_clean_cache(
) RETURNS BOOL
AS
'MODULE_PATHNAME',
'ns_clean_cache'
    LANGUAGE C STRICT
               STABLE;
