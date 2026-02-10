package storage

import "testing"

func TestNewStoreMemory(t *testing.T) {
	store, err := NewStore("memory", "")
	if err != nil {
		t.Fatalf("new memory store: %v", err)
	}
	if store == nil {
		t.Fatal("expected non-nil store")
	}
}

func TestNewStoreUnsupported(t *testing.T) {
	_, err := NewStore("unknown", "")
	if err == nil {
		t.Fatal("expected unsupported store error")
	}
}
