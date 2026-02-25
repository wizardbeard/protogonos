package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunDataExtractGTSA(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw.csv")
	out := filepath.Join(tmp, "gtsa.csv")
	raw := "t,close,vol\n0,1.1,10\n1,1.2,11\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "gtsa",
		"--in", in,
		"--out", out,
		"--value-col", "close",
	})
	if err != nil {
		t.Fatalf("run data-extract gtsa: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if got != "t,value\n0,1.1\n1,1.2\n" {
		t.Fatalf("unexpected gtsa output:\n%s", got)
	}
}

func TestRunDataExtractEpitopes(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw_ep.csv")
	out := filepath.Join(tmp, "ep.csv")
	raw := "signal,memory,class,aa0,aa1\n0.1,0.0,1,4,10\n-0.2,0.1,0,2,2\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "epitopes",
		"--in", in,
		"--out", out,
		"--sequence-cols", "aa0,aa1",
	})
	if err != nil {
		t.Fatalf("run data-extract epitopes: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if !strings.Contains(got, "signal,memory,class,seq0,seq1\n") {
		t.Fatalf("unexpected epitopes header:\n%s", got)
	}
}
