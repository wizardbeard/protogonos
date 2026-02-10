//go:build sqlite

package storage

func newSQLiteStore(path string) (Store, error) {
	return NewSQLiteStore(path), nil
}
