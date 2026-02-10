//go:build sqlite

package storage

func DefaultStoreKind() string {
	return "sqlite"
}
