# NRIndex Index Access Method Design

## Overview

The `nrindex` index access method is a PostgreSQL extension that uses RocksDB as the storage backend for index data. This document describes the design and implementation of the index access method.

## Key Design Decisions

### 1. Separate Index Key-Value Structures

**Problem**: The original implementation incorrectly reused `NRAMKey` and `NRAMValue` structures, which are designed for table storage, not index storage.

**Solution**: Created dedicated index-specific structures:

- **NRIndexKey**: Contains the actual indexed column values serialized
- **NRIndexValue**: Contains references to heap tuples (ItemPointer) and transaction metadata

### 2. Index Key Structure (NRIndexKey)

```c
typedef struct NRIndexKeyData {
    Oid indexOid;              /* Index relation OID */
    uint32 key_size;           /* Size of serialized key data */
    char key_data[FLEXIBLE_ARRAY_MEMBER];  /* Serialized index key values */
} NRIndexKeyData;
```

**Purpose**: 
- Stores the actual indexed column values in serialized form
- Includes index OID for namespace separation
- Variable-length structure to handle different data types

### 3. Index Value Structure (NRIndexValue)

```c
typedef struct NRIndexValueData {
    ItemPointerData heap_tid;  /* Reference to heap tuple */
    TransactionId xact_id;     /* Transaction that created this index entry */
    uint16 flags;              /* Flags for index entry state */
} NRIndexValueData;
```

**Purpose**:
- Stores reference to the heap tuple (ItemPointer)
- Includes transaction information for MVCC
- Flags for entry state (private, deleted, etc.)

## Architecture

### 1. Storage Layer Integration

The index access method integrates with the existing RocksDB infrastructure:

- Uses `RocksClientPut()`, `RocksClientGet()`, `RocksClientDelete()`, `RocksClientRangeScan()`
- Converts between index structures and NRAM structures for storage
- Maintains compatibility with existing transaction and locking mechanisms

### 2. Key Serialization

Index keys are serialized using PostgreSQL's datum serialization functions:

- Handles variable-length data types (varlena)
- Supports all PostgreSQL data types
- Maintains proper ordering for range scans

### 3. Scan Operations

The index supports:

- **Point lookups**: Direct key-value retrieval
- **Range scans**: Iteration over key ranges
- **Bitmap scans**: Building TID bitmaps for bitmap heap scans
- **NULL handling**: Special handling for NULL values (TODO)

## Implementation Files

### Core Files

1. **`src/nrindex.c`**: Main index access method implementation
2. **`src/nrindex.h`**: Public interface declarations
3. **`src/nrindex_access/nrindex_kv.h`**: Index key-value structure definitions
4. **`src/nrindex_access/nrindex_kv.c`**: Index key-value implementation

### Integration Files

1. **`sql/nram--1.0.sql`**: SQL extension definitions
2. **`Makefile`**: Build configuration
3. **`nram.control`**: Extension metadata

## Key Functions

### Index Operations

- `nrindex_build()`: Build index from heap relation
- `nrindex_insert()`: Insert new index entry
- `nrindex_bulkdelete()`: Bulk deletion of index entries
- `nrindex_vacuumcleanup()`: Post-vacuum cleanup

### Scan Operations

- `nrindex_beginscan()`: Initialize index scan
- `nrindex_rescan()`: Restart scan with new conditions
- `nrindex_gettuple()`: Get next tuple from scan
- `nrindex_getbitmap()`: Build TID bitmap
- `nrindex_endscan()`: Cleanup scan resources

### Key-Value Operations

- `nrindex_key_create()`: Create index key from column values
- `nrindex_value_create()`: Create index value from heap TID
- `nrindex_rocks_get()`: Retrieve index entry
- `nrindex_rocks_put()`: Store index entry
- `nrindex_rocks_range_scan()`: Perform range scan

## Usage Example

```sql
-- Create extension
CREATE EXTENSION nram;

-- Create table with nrindex index
CREATE TABLE test_table (
    id INTEGER PRIMARY KEY,
    name TEXT,
    value INTEGER
) USING nram;

CREATE INDEX test_idx ON test_table (name, value) USING nrindex;

-- Use index in queries
SELECT * FROM test_table WHERE name = 'test' AND value > 100;
```

## Future Improvements

1. **NULL Handling**: Complete implementation of NULL value indexing
2. **Unique Constraints**: Enhanced unique constraint enforcement
3. **Partial Indexes**: Support for partial index predicates
4. **Concurrent Index Build**: Parallel index building
5. **Index-Only Scans**: Return data directly from index when possible
6. **Statistics**: Collect and maintain index statistics
7. **Compression**: Index data compression for storage efficiency

## Testing

The implementation includes:

- Unit tests for key-value operations
- Integration tests with PostgreSQL's index testing framework
- Performance benchmarks against standard PostgreSQL indexes

## Compatibility

- **PostgreSQL Version**: Compatible with PostgreSQL 15+
- **RocksDB Version**: Requires RocksDB 6.0+
- **Platforms**: Linux, macOS (Windows support planned)
