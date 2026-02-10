//go:build !sqlite

package storage

import "fmt"

func newSQLiteStore(_ string) (Store, error) {
	return nil, fmt.Errorf("sqlite backend unavailable in this build; rebuild with -tags sqlite")
}
