package storage

import (
	"FC/configs"
	"fmt"
	"sync/atomic"
)

type Table struct {
	tableName             string
	attributesNum         int
	autoIncreasingPrimary int32
}

func (tab *Table) GenPrimaryKey() int {
	return int(atomic.AddInt32(&tab.autoIncreasingPrimary, 1))
}

type Key uint64
type Access uint8

type RowData struct {
	Length    uint          `bson:"length"`
	Value     []interface{} `bson:"value"`
	fromTable *Table
}

//type Manager struct {
//	Lock LockManager
//}

type RowRecord struct {
	Epoch      uint64
	RowID      Key
	FromTable  *Table
	PrimaryKey Key
	Data       *RowData // attributes get merged into one string
}

func NewRowData(tb *Table) *RowData {
	res := &RowData{}
	res.fromTable = tb
	if tb != nil {
		res.Length = uint(tb.attributesNum)
		res.Value = make([]interface{}, tb.attributesNum)
	} else {
		res.Length = 1
		res.Value = make([]interface{}, 1)
	}
	return res
}

func NewRowDataWithLength(len int) *RowData {
	res := &RowData{}
	res.fromTable = nil
	res.Length = uint(len)
	res.Value = make([]interface{}, len)
	return res
}

func NewRowRecord(table *Table, primaryKey Key, rowKey Key) *RowRecord {
	res := &RowRecord{
		RowID:      rowKey,
		FromTable:  table,
		PrimaryKey: primaryKey,
		Data:       NewRowData(table),
	}
	return res
}

func (r *RowRecord) Copy(row *RowRecord) {
	r.Data = row.Data
	r.RowID = row.RowID
	r.FromTable = row.FromTable
	r.Epoch = row.Epoch
	r.PrimaryKey = row.PrimaryKey
}

func (r *RowRecord) CopyData(row *RowRecord) {
	r.Data = row.Data
}

func (c *RowData) SetAttribute(idx uint, value interface{}) {
	configs.Assert(idx < c.Length && idx >= 0, "attribute access out of range")
	c.Value[idx] = value
}

func (c *RowData) GetAttribute(idx uint) interface{} {
	configs.Assert(idx < c.Length && idx >= 0, "attribute access out of range")
	return c.Value[idx]
}

func (c *RowData) String() string {
	//return "Dummy data"
	return fmt.Sprintf("%v", c.Value[0])
}

func (r *RowRecord) GetValue(idx uint) interface{} {
	return r.Data.GetAttribute(idx)
}

func (r *RowRecord) SetValue(idx uint, value interface{}) {
	r.Data.SetAttribute(idx, value)
}
