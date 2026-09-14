// Unified Plan Buffer schema for dbdiagram.io

Table plan_buffer {
  id integer [primary key, increment]
  query_hash text [not null]           // hash/id of the SQL
  query text [not null]                // raw SQL string
  actual_plan_json text [not null]     // EXPLAIN(ANALYZE,FORMAT JSON) result
  actual_plan_hash text                // hash of plan structure for dedup/analysis
  actual_latency real                  // measured latency if available
  plan_time real                       // planning time (ms) if available
  hint_json text                       // JSON array of hint statements (e.g., SET ...), optional
  join_order_hint text                 // leading hint string, optional

  indexes {
    (query_hash)
    (actual_plan_hash)
    (actual_latency)
  }
}