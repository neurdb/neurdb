-- nram--1.0.sql
CREATE FUNCTION nram_tableam_handler(internal)
RETURNS table_am_handler
AS 'nram', 'nram_tableam_handler'
LANGUAGE C STRICT;

CREATE ACCESS METHOD nram TYPE TABLE HANDLER nram_tableam_handler;

CREATE OR REPLACE FUNCTION nram_load_policy(path text)
RETURNS void
AS 'nram', 'nram_load_policy'
LANGUAGE C
STRICT
VOLATILE
PARALLEL RESTRICTED;

CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;


-- nrindex index access method
CREATE FUNCTION nrindex_handler(internal)
RETURNS index_am_handler
AS 'nram', 'nrindex_handler'
LANGUAGE C STRICT;

CREATE ACCESS METHOD nrindex TYPE INDEX HANDLER nrindex_handler;

-- Operator class for INT4 (integer)
CREATE OPERATOR FAMILY nrindex_int4_ops USING nrindex;
CREATE OPERATOR CLASS int4_nrindex_ops
    DEFAULT FOR TYPE int4 USING nrindex
    FAMILY nrindex_int4_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 btint4cmp(int4, int4);

-- Operator class for INT8 (bigint)
CREATE OPERATOR FAMILY nrindex_int8_ops USING nrindex;
CREATE OPERATOR CLASS int8_nrindex_ops
    DEFAULT FOR TYPE int8 USING nrindex
    FAMILY nrindex_int8_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 btint8cmp(int8, int8);