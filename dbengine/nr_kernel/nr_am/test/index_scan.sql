DROP TABLE IF EXISTS xs CASCADE;
CREATE TABLE xs (
                    id  INT,
                    tag TEXT,
                    val TEXT,
                    PRIMARY KEY (id)
) USING nram;

CREATE INDEX xs_tag_idx ON xs(tag);

-- Insert dummy data.
INSERT INTO xs
SELECT i, 'tag' || (i % 10), 'val' || i
FROM generate_series(1, 50) AS i;

SET enable_seqscan = OFF;

-- Test equality scan on primary key.
EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, TIMING OFF)
SELECT * FROM xs WHERE id = 42;

SELECT * FROM xs WHERE id = 42;

-- Test equality scan on secondary index.
EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, TIMING OFF)
SELECT * FROM xs WHERE tag = 'tag3';

SELECT * FROM xs WHERE tag = 'tag3';

-- Test range scan on primary key.
EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, TIMING OFF)
SELECT * FROM xs WHERE id > 45;

SELECT * FROM xs WHERE id > 45;

-- Cleanup.
DROP TABLE xs;
SET enable_seqscan = ON;
