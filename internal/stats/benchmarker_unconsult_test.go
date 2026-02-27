package stats

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteBenchmarkerUnconsult(t *testing.T) {
	outPath := filepath.Join(t.TempDir(), "benchmarks", "alife_benchmark")
	items := []any{
		"exp-run-001",
		map[string]any{"run_id": "exp-run-002", "final_best": 0.7},
	}
	if err := WriteBenchmarkerUnconsult(outPath, items); err != nil {
		t.Fatalf("write benchmarker unconsult: %v", err)
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read benchmarker unconsult: %v", err)
	}
	text := string(data)
	if !strings.Contains(text, "\"exp-run-001\"") {
		t.Fatalf("expected first item line, got:\n%s", text)
	}
	if !strings.Contains(text, "\"run_id\":\"exp-run-002\"") {
		t.Fatalf("expected second item line, got:\n%s", text)
	}
	if strings.Count(text, "\n") != 2 {
		t.Fatalf("expected 2 lines, got %d", strings.Count(text, "\n"))
	}
}
