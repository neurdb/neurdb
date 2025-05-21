-- nram--1.0.sql
CREATE FUNCTION nram_tableam_handler(internal)
RETURNS table_am_handler
AS 'nram', 'nram_tableam_handler'
LANGUAGE C STRICT;

CREATE ACCESS METHOD nram TYPE TABLE HANDLER nram_tableam_handler;
