/* nr_kernel/nr_molqo/sql/nr_molqo--1.0.sql */

-- NeurDB MoLQO (Mixture of Learned Query Optimizer) Extension
-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION nr_molqo" to load this file. \quit

-- Function to manually test MoLQO optimization
CREATE FUNCTION optimize_query(text)
RETURNS text
AS 'MODULE_PATHNAME'
LANGUAGE C STRICT;

-- Function to get MoLQO status
CREATE FUNCTION molqo_status()
RETURNS TABLE (
    setting text,
    value text,
    description text
)
AS $$
SELECT 
    unnest(ARRAY['enable_molqo', 'molqo.server_url']),
    unnest(ARRAY[
        current_setting('enable_molqo', true),
        current_setting('molqo.server_url', true)
    ]),
    unnest(ARRAY[
        'Enable MoLQO mixture of learned query optimizer',
        'MoLQO server URL'
    ])
$$ LANGUAGE SQL;
