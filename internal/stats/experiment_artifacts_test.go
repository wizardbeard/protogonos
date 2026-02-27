package stats

import "testing"

func TestWriteReadAndListBenchmarkExperiments(t *testing.T) {
	base := t.TempDir()
	expA := BenchmarkExperiment{
		ID:           "exp-a",
		Notes:        "first",
		ProgressFlag: "in_progress",
		RunIndex:     2,
		TotalRuns:    5,
		StartedAtUTC: "2026-02-27T00:00:00Z",
	}
	expB := BenchmarkExperiment{
		ID:           "exp-b",
		Notes:        "second",
		ProgressFlag: "completed",
		RunIndex:     6,
		TotalRuns:    5,
		StartedAtUTC: "2026-02-28T00:00:00Z",
	}
	if err := WriteBenchmarkExperiment(base, expA); err != nil {
		t.Fatalf("write exp a: %v", err)
	}
	if err := WriteBenchmarkExperiment(base, expB); err != nil {
		t.Fatalf("write exp b: %v", err)
	}

	read, ok, err := ReadBenchmarkExperiment(base, "exp-a")
	if err != nil {
		t.Fatalf("read exp a: %v", err)
	}
	if !ok {
		t.Fatalf("expected exp a to exist")
	}
	if read.ID != "exp-a" || read.RunIndex != 2 {
		t.Fatalf("unexpected exp a payload: %+v", read)
	}

	list, err := ListBenchmarkExperiments(base)
	if err != nil {
		t.Fatalf("list experiments: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("expected 2 experiments, got %d", len(list))
	}
	if list[0].ID != "exp-b" || list[1].ID != "exp-a" {
		t.Fatalf("unexpected list ordering: %+v", list)
	}
}
