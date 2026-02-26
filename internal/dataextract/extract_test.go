package dataextract

import (
	"strings"
	"testing"
)

func TestExtractSeriesCSVByHeaderName(t *testing.T) {
	in := strings.NewReader("t,close,volume\n0,1.01,10\n1,1.02,11\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,1.01\n1,1.02\n" {
		t.Fatalf("unexpected series output:\n%s", got)
	}
}

func TestExtractEpitopesCSVWithNamedSequenceColumns(t *testing.T) {
	in := strings.NewReader("signal,memory,class,aa0,aa1\n0.1,0.0,1,4,10\n-0.2,0.1,0,2,2\n")
	var out strings.Builder
	err := ExtractEpitopesCSV(in, &out, EpitopesOptions{
		HasHeader:           true,
		SignalColumnName:    "signal",
		MemoryColumnName:    "memory",
		ClassColumnName:     "class",
		SequenceColumnNames: []string{"aa0", "aa1"},
		SignalColumnIndex:   -1,
		MemoryColumnIndex:   -1,
		ClassColumnIndex:    -1,
	})
	if err != nil {
		t.Fatalf("extract epitopes: %v", err)
	}
	if got := out.String(); got != "signal,memory,class,seq0,seq1\n0.1,0,1,4,10\n-0.2,0.1,0,2,2\n" {
		t.Fatalf("unexpected epitopes output:\n%s", got)
	}
}

func TestExtractSeriesCSVNormalizeMinMax(t *testing.T) {
	in := strings.NewReader("close\n10\n20\n30\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
		Normalize:         "minmax",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,0\n1,0.5\n2,1\n" {
		t.Fatalf("unexpected minmax output:\n%s", got)
	}
}

func TestExtractSeriesCSVNormalizeZScore(t *testing.T) {
	in := strings.NewReader("close\n10\n20\n30\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
		Normalize:         "zscore",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,-1.224744871391589\n1,0\n2,1.224744871391589\n" {
		t.Fatalf("unexpected zscore output:\n%s", got)
	}
}
