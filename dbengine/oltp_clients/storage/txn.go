package storage

import (
	"FC/configs"
	"context"
	"github.com/jackc/pgx/v4"
	"github.com/viney-shih/go-lock"
)

const (
	txnExecution = 0
	txnPrepare   = 1
	txnPreCommit = 2
	txnCommitted = 3
	txnAborted   = 4
)

const (
	TxnScan     = 0
	TxnRead     = 1
	TxnWrite    = 2
	TxnRollBack = 3
)

type txnAccess struct {
	Origin     *RowRecord
	Local      *RowRecord
	AccessType uint8
}

type DBTxn struct {
	latch     lock.Mutex
	lockReady bool
	lockAbort bool

	BeginTS  uint32
	CommitTS uint32
	TxnState uint8
	Finished bool

	AbortCnt  int
	RowCnt    int
	WriteCnt  int
	Accesses  []*txnAccess
	LogBuffer []*RedoLogEntry

	insertRows []*RowRecord
	timestamp  uint32
	txnID      uint32
	writeCopy  bool

	sqlTX             pgx.Tx
	isPrepared        bool
	mdbPendingLatches map[string]uint8
}

func NewTxn(ctx context.Context) *DBTxn {
	res := &DBTxn{}
	res.latch = lock.NewCASMutex()
	res.RowCnt = 0
	res.WriteCnt = 0
	res.AbortCnt = 0
	res.TxnState = txnExecution
	res.LogBuffer = make([]*RedoLogEntry, configs.MaxAccessesPerTxn, configs.MaxAccessesPerTxn)
	res.Accesses = make([]*txnAccess, configs.MaxAccessesPerTxn, configs.MaxAccessesPerTxn)
	if ctx.Value("store").(string) == configs.MongoDB {
		res.mdbPendingLatches = make(map[string]uint8)
	}
	return res
}

func (c *DBTxn) SetTxnID(tid uint32) {
	c.txnID = tid
}

func (c *DBTxn) GetTxnID() uint32 {
	return c.txnID
}

func (c *DBTxn) SetTS(tid uint32) {
	c.timestamp = tid
}

func (c *DBTxn) GetTS() uint32 {
	return c.timestamp
}

// InsertRow2Index TODO: support index lock, following the Slog.
func (c *DBTxn) InsertRow2Index(row *RowRecord) error {
	configs.Assert(configs.DeferredInsert, "we have not implemented the non-deferred insert, rollback requires index delete")
	c.insertRows = append(c.insertRows, row)
	return nil
}

func (c *DBTxn) Finish(isCommit bool) {
	// TODO: when implementing OCC, need to perform validate here.
	if isCommit {
		c.TxnState = txnCommitted
	} else {
		c.TxnState = txnAborted
	}
}

// TryFinish only one can succeed in finish.
func (c *DBTxn) TryFinish() bool {
	if c.Finished {
		return false
	} else {
		c.Finished = true
		return true
	}
}
