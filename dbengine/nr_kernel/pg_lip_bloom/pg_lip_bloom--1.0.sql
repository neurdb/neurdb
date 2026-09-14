CREATE FUNCTION pg_lip_bloom_init(integer) RETURNS integer
AS 'MODULE_PATHNAME', 'pg_lip_bloom_init'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION pg_lip_bloom_set_dynamic(integer) RETURNS integer
AS 'MODULE_PATHNAME', 'pg_lip_bloom_set_dynamic'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION pg_lip_bloom_add(integer, integer) RETURNS integer
AS 'MODULE_PATHNAME', 'pg_lip_bloom_add'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION pg_lip_bloom_probe(integer, integer) RETURNS boolean
AS 'MODULE_PATHNAME', 'pg_lip_bloom_probe'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE
COST 1000;

CREATE FUNCTION pg_lip_bloom_info() RETURNS integer
AS 'MODULE_PATHNAME', 'pg_lip_bloom_info'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON EXTENSION pg_lip_bloom IS
'Runtime Bloom filters used by NQO LIP actions';
