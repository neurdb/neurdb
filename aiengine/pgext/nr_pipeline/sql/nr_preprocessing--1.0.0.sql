CREATE FUNCTION nr_inference(model_name text, model_id int, table_name text, batch_size int, batch_num int, nfeat int, columns text[])
    RETURNS SETOF RECORD
AS
'MODULE_PATHNAME',
'nr_inference'
    LANGUAGE C STRICT
               VOLATILE;


CREATE FUNCTION nr_train(model_name text, table_name text, batch_size int, batch_num int, epoch int, nfeat int, columns text[], target text)
    RETURNS BOOL
AS 'MODULE_PATHNAME',
'nr_train'
    LANGUAGE C STRICT
               VOLATILE;


CREATE FUNCTION nr_finetune(model_name text, model_id int, table_name text, batch_size int, batch_num int, epoch int, nfeat int, columns text[], target text)
    RETURNS BOOL
AS 'MODULE_PATHNAME',
'nr_finetune'
    LANGUAGE C STRICT
               VOLATILE;


CREATE FUNCTION nr_model_lookup(table_name text, features text[], target text)
    RETURNS INT
AS 'MODULE_PATHNAME',
'nr_model_lookup'
    LANGUAGE C STRICT
               VOLATILE;
