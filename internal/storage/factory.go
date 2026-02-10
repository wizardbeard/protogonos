package storage

import "fmt"

func NewStore(kind, sqlitePath string) (Store, error) {
	switch kind {
	case "", "memory":
		return NewMemoryStore(), nil
	case "sqlite":
		return newSQLiteStore(sqlitePath)
	default:
		return nil, fmt.Errorf("unsupported store backend: %s", kind)
	}
}

func CloseIfSupported(store Store) error {
	closer, ok := store.(interface{ Close() error })
	if !ok {
		return nil
	}
	return closer.Close()
}
