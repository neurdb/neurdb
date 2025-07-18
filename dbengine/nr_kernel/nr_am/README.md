# NRAM Table Access Method

## 1. Build the NRAM TAM

```bash
make clean
make
sudo make install
````

To run unit tests:

```bash
make test
```

> **Note:** This extension depends on **RocksDB version 10.3.0**.

---

## 2. Load the NRAM Extension in PostgreSQL

```sql
CREATE EXTENSION nram;
```

---

## 3. Verify That the Handler Function Is Loaded

```sql
SELECT * FROM pg_proc WHERE proname = 'nram_tableam_handler';
```
