package storage

import (
	"FC/configs"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"sync"
	"time"
)

// Shard maintains a Local kv-store and all information needed.
type Shard struct {
	shardID                 string
	mu                      *sync.Mutex
	txnPool                 sync.Map
	ctx                     context.Context
	LockWindowInjectedDelay time.Duration

	// In case of benchmark
	length           int
	tables           sync.Map // tables with a primary index for each table.
	secondaryIndexes sync.Map
	//concurrentTxnNum uint32
	log *LogManager

	// In case of MongoDB
	mdb *MongoDB

	// In case of PostgreSQL.
	db *SQLDB
	// In case of MySQL

	// In case of ElasticSearch

	// In case of GridFS

	// In case of GCS
}

func (c *Shard) GetID() string {
	return c.shardID
}

// AddTable add a new table into the shard.
func (c *Shard) AddTable(tableName string, attributeNum int) *Table {
	tab := &Table{tableName: tableName, attributesNum: attributeNum, autoIncreasingPrimary: 0}
	c.tables.Store(tableName, tab)
	return tab
}

// Clear all pending transactions and connections.
func (c *Shard) Clear() {
	switch c.ctx.Value("store").(string) {
	case configs.PostgreSQL:
		//c.db.mustExec("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid != pg_backend_pid();")
		rows, err := c.db.pool.Query(c.db.ctx, "SELECT gid FROM pg_prepared_xacts;")
		if err != nil {
			panic(err)
		}
		for rows.Next() {
			var gid string
			if err := rows.Scan(&gid); err != nil {
				log.Fatal(err)
			}
			_, err := c.db.pool.Exec(c.db.ctx, fmt.Sprintf("ROLLBACK PREPARED '%s'", gid))
			if err != nil {
				log.Printf("Failed to rollback prepared transaction with gid %s: %v", gid, err)
			} else {
				fmt.Printf("Rolled back prepared transaction with gid %s\n", gid)
			}
		}
	default:
		// nothing to do.
	}
}

func newShardKV(shardID string, storeType string, delay time.Duration) *Shard {
	c := &Shard{
		shardID:                 shardID,
		mu:                      &sync.Mutex{},
		ctx:                     context.WithValue(context.Background(), "store", storeType),
		LockWindowInjectedDelay: delay,
		log:                     NewLogManager(shardID),
	}
	if c.ctx.Value("store").(string) == configs.MongoDB {
		c.mdb = &MongoDB{}
		c.mdb.init(shardID[len(shardID)-3:])
	} else if c.ctx.Value("store").(string) == configs.PostgreSQL {
		c.db = &SQLDB{}
		c.db.init()
		c.Clear()
	} else {
		panic("not supported")
	}
	return c
}

/* Interactive simple key-Value APIs.*/
// TODO: support delete and range scan with lock.

func (c *Shard) Insert(tableName string, key uint64, value *RowData) bool {
	if c.ctx.Value("store").(string) == configs.MongoDB {
		if !c.mdb.Insert(tableName, key, value) {
			return false
		} else {
			return true
		}
	} else if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.db.Insert(tableName, key, value)
	} else {
		panic("not supported")
	}
}

func (c *Shard) Update(tableName string, key uint64, value *RowData) bool {
	if c.ctx.Value("store").(string) == configs.MongoDB {
		return c.mdb.Update(tableName, key, value)
	} else if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.db.Update(tableName, key, value)
	} else {
		panic("not supported")
	}
}

func (c *Shard) Read(tableName string, key uint64) (*RowData, bool) {
	if c.ctx.Value("store").(string) == configs.MongoDB {
		return c.mdb.Read(tableName, key)
	} else if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.db.Read(tableName, key)
	} else {
		panic("not supported")
	}
}

/* Execution phase APIs for transactions. */

func (c *Shard) Begin(txnID uint32, opt []TXOpt) bool {
	configs.TPrintf("TXN" + strconv.FormatUint(uint64(txnID), 10) + ": transaction begun")
	_, ok := c.txnPool.Load(txnID)
	configs.Assert(!ok, "the previous transaction has not been finished yet (TID="+strconv.Itoa(int(txnID))+")")
	txn := NewTxn(c.ctx)
	txn.latch.Lock()
	defer txn.latch.Unlock()
	txn.txnID = txnID
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		var err error
		txn.sqlTX, err = c.db.Begin(configs.DefaultIsolationLevel)
		if err != nil {
			panic(err)
		}
		txn.isPrepared = false
		if err != nil {
			panic(err)
		}
	}
	c.txnPool.Store(txnID, txn)
	return true
}

func JPrint(v interface{}) {
	byt, _ := json.Marshal(v)
	fmt.Println(string(byt))
}

func (c *Shard) ReadTxn(tableName string, txnID uint32, key uint64) (*RowData, bool) {
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.ReadTxnPostgres(tableName, txnID, key)
	} else {
		panic("not supported")
	}
}

func (c *Shard) UpdateTxn(tableName string, txnID uint32, key uint64, value *RowData) bool {
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.UpdateTxnPostgres(tableName, txnID, key, value)
	} else {
		panic("not supported")
	}
}

func (c *Shard) RollBack(txnID uint32) bool {
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.RollBackPostgres(txnID)
	} else {
		panic("not supported")
	}
}

func (c *Shard) Commit(txnID uint32) bool {
	configs.TimeTrack(time.Now(), fmt.Sprintf("commit on shard %s", c.shardID), uint64(txnID))
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		return c.CommitPostgres(txnID)
	} else {
		panic("not supported")
	}
}

/* APIs for distributed transactions. */

func (c *Shard) Prepare(txnID uint32) bool {
	v, ok := c.txnPool.Load(txnID)
	if !ok {
		configs.Warn(ok, "the transaction has been aborted.")
		// this could happen when the transaction exceeds crash timeout, the coordinator has asserted abort.
		return false
	}
	tx := v.(*DBTxn)
	tx.latch.Lock()
	defer tx.latch.Unlock()
	time.Sleep(c.LockWindowInjectedDelay)
	configs.Assert(ok, "the transaction has finished before commit on this node.")
	if tx.TxnState != txnExecution {
		// in G-PAC, this is a possible corner case that the transaction get committed/aborted before pre-write on a replica.
		return false
	}
	configs.Assert(tx.TxnState == txnExecution, "the transaction shall be in execution state before")
	configs.Assert(tx.txnID == txnID, "different transaction running")
	tx.TxnState = txnPrepare
	c.log.writeRedoLog4Txn(tx)
	c.log.writeTxnState(tx)
	if c.ctx.Value("store").(string) == configs.PostgreSQL {
		_, err := tx.sqlTX.Exec(c.db.ctx, fmt.Sprintf("PREPARE TRANSACTION 'TXN_%v_%v'", txnID, c.shardID))
		if err != nil {
			return false
		}
		tx.isPrepared = true
	} else {
		panic("not supported")
	}
	return true
}

func (c *Shard) PreCommit(txnID uint32) bool {
	v, ok := c.txnPool.Load(txnID)
	if !ok {
		configs.Warn(ok, "the transaction has been aborted.")
		return false
	}
	tx := v.(*DBTxn)
	tx.latch.Lock()
	defer tx.latch.Unlock()
	configs.Assert(ok, "the transaction has finished before commit on this node.")
	configs.Assert(tx.txnID == txnID, "different transaction running")
	//configs.JPrint(tx)
	configs.Assert(tx.TxnState == txnPrepare, "in 3PC, a transaction branch shall be prepared before pre-commit")
	tx.TxnState = txnPreCommit
	c.log.writeTxnState(tx)
	return true
}

// PreCommitAsync it is possible for a replica to receive pre-commit message before receiving prepare message.
// in this ignore it and does not reply the ACK.
func (c *Shard) PreCommitAsync(txnID uint32) bool {
	v, ok := c.txnPool.Load(txnID)
	if !ok {
		configs.Warn(ok, "the transaction has been aborted.")
		// the aborted transaction can also participant in the accepting process.
		return true
	}
	tx := v.(*DBTxn)
	tx.latch.Lock()
	defer tx.latch.Unlock()
	configs.Assert(ok, "the transaction has finished before commit on this node.")
	configs.Assert(tx.txnID == txnID, "different transaction running")
	if tx.TxnState != txnPrepare {
		return false
	}
	configs.Assert(tx.TxnState == txnPrepare, "in 3PC, a transaction branch shall be prepared before pre-commit")
	tx.TxnState = txnPreCommit
	c.log.writeTxnState(tx)
	return true
}
