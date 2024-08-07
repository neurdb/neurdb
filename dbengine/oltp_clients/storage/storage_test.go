package storage

import (
	"FC/configs"
	"fmt"
	"github.com/magiconair/properties/assert"
	"math/rand"
	"testing"
	"time"
)

func TestNoContentionWrite(t *testing.T) {
	s := Testkit("id", configs.PostgreSQL)
	st := time.Now()
	for i := 0; i < 100000; i++ {
		ok := s.Update("MAIN", uint64(rand.Intn(1000)+1), s.GenTestValue())
		assert.Equal(t, ok, true)
	}
	fmt.Println("No contention write/second = ", 100000.0/float64(time.Since(st).Seconds()))
}

func TestNoContentionRead(t *testing.T) {
	s := Testkit("id", configs.PostgreSQL)
	st := time.Now()
	for i := 0; i < 100000; i++ {
		key := uint64(rand.Intn(1000) + 1)
		v, ok := s.Read("MAIN", key)
		assert.Equal(t, ok, true)
		assert.Equal(t, int(key+3), v.GetAttribute(0).(int))
	}
	fmt.Println("No contention read/second = ", 100000.0/float64(time.Since(st).Seconds()))
}

func TestW4R(t *testing.T) {
	s := Testkit("id", configs.PostgreSQL)
	go func() {
		for i := 0; i < 100000; i++ {
			s.Read("MAIN", uint64(rand.Intn(100)+1))
		}
	}()
	go func() {
		for i := 0; i < 1000000; i++ {
			s.Read("MAIN", uint64(rand.Intn(100)+1))
		}
	}()
	st := time.Now()
	for i := 0; i < 100000; i++ {
		s.Update("MAIN", uint64(rand.Intn(1000)+1), s.GenTestValue())
	}
	fmt.Println("Write/second with two thread accessing = ", 100000.0/float64(time.Since(st).Seconds()))
}
