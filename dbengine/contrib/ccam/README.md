1. Build the TAM.

```bash
make clean
make
sudo make install
```

2. Load the CCAM.


```sql
CREATE OR REPLACE FUNCTION ccam_tableam_handler(internal)
RETURNS table_am_handler
AS 'ccam', 'ccam_tableam_handler'
LANGUAGE C STRICT;


CREATE ACCESS METHOD ccam TYPE TABLE HANDLER ccam_tableam_handler;
```

2. Verify that the handler function has been successfully loaded.

```sql
SELECT * FROM pg_proc WHERE proname = 'ccam_tableam_handler';
```