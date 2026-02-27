package stats

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

func WriteBenchmarkerUnconsult(path string, items []any) error {
	if path == "" {
		return fmt.Errorf("output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, item := range items {
		data, err := json.Marshal(item)
		if err != nil {
			return err
		}
		if _, err := file.Write(data); err != nil {
			return err
		}
		if _, err := file.Write([]byte("\n")); err != nil {
			return err
		}
	}
	return nil
}
