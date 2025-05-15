1. Build the TAM.

```bash
make clean
make
sudo make install
```

2. Load the NRAM.


```sql
CREATE OR REPLACE FUNCTION nram_tableam_handler(internal)
RETURNS table_am_handler
AS 'nram', 'nram_tableam_handler'
LANGUAGE C STRICT;


CREATE ACCESS METHOD nram TYPE TABLE HANDLER nram_tableam_handler;
```

2. Verify that the handler function has been successfully loaded.

```sql
SELECT * FROM pg_proc WHERE proname = 'nram_tableam_handler';
```
