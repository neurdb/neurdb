# Issue: lws buflist sanity limit during training (WebSocket send buffer overflow)

## Summary

During streaming training, the C-side executor logs repeated errors:

```
lws_buflist_append_segment: buflist reached sanity limit
```

This comes from **libwebsockets (lws)**. The send-side buffer list (buflist) hits an internal sanity limit because data is being queued faster than it can be sent.

## Root cause

- The C side pushes training batches to the AI server over WebSocket via `nws_send_batch_data()` (e.g. in `interface2.c` / `interface.c`).
- Each batch is sent with `lws_write()`; lws buffers outgoing data in a buflist.
- The executor produces batches quickly (e.g. one per 512 rows), while the Python side consumes them slowly (training per batch takes seconds).
- There is **no back pressure**: C keeps appending segments to the buflist without waiting for the socket to drain or for the consumer to acknowledge.
- The buflist grows until lws hits its sanity limit and logs the error repeatedly.

## Impact

- Log spam and possible instability when many batches are sent in quick succession.
- Training may appear "stuck" because the Python side is waiting for the next batch while the C/WebSocket layer is overloaded or misbehaving.

## Proposed direction

1. **Back pressure (recommended)**  
   - Do not send the next training batch until the previous one has been consumed or the send path has room.  
   - Options: (a) protocol change so Python signals "ready for next batch", or (b) bounded send queue on the C side that blocks when full until lws drains.

2. **Throttling**  
   - After each `nws_send_batch_data()`, run `lws_service` (and/or `lws_callback_on_writable`) so that queued data is drained before sending the next batch. This reduces buflist buildup without a protocol change.

3. **Increase lws sanity limit (mitigation only)**  
   - If lws is built from source, increase the buflist sanity limit. This only postpones the issue and may increase memory use; back pressure or throttling is still preferred.

## Relevant code

- WebSocket send: `dbengine/nr_kernel/nr_pipeline/src/utils/network/websocket.c`  
  - `nws_send_batch_data()`, `send_json()`, `lws_write()`
- Call sites that send training batches:  
  - `dbengine/nr_kernel/nr_pipeline/src/interface2.c` (e.g. `nws_send_batch_data(..., S_TRAIN, ...)`)  
  - `dbengine/nr_kernel/nr_pipeline/src/interface.c` (same)

## Labels (suggested)

- component: dbengine / nr_pipeline / websocket  
- type: bug  
- priority: medium (log spam + possible training flow impact)
