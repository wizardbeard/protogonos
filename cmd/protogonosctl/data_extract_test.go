package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/dataextract"
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

func TestRunDataExtractGTSAWithNormalization(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw_norm.csv")
	out := filepath.Join(tmp, "gtsa_norm.csv")
	raw := "close\n10\n20\n30\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "gtsa",
		"--in", in,
		"--out", out,
		"--value-col", "close",
		"--normalize", "minmax",
	})
	if err != nil {
		t.Fatalf("run data-extract gtsa normalize: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if got != "t,value\n0,0\n1,0.5\n2,1\n" {
		t.Fatalf("unexpected normalized gtsa output:\n%s", got)
	}
}

func TestRunDataExtractMNIST(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "mnist_raw.csv")
	out := filepath.Join(tmp, "mnist.csv")
	raw := "px0,px1,label\n0,255,0\n128,64,9\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "mnist",
		"--in", in,
		"--out", out,
		"--label-col", "label",
	})
	if err != nil {
		t.Fatalf("run data-extract mnist: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if !strings.Contains(got, "class9,class8,class7,class6,class5,class4,class3,class2,class1,class0\n") {
		t.Fatalf("unexpected mnist header:\n%s", got)
	}
}

func TestRunDataExtractChrHMM(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "chr_hmm_raw.csv")
	out := filepath.Join(tmp, "chr_hmm.csv")
	raw := "chr,from,to,tag,a,b\nchr22,100,200,Enh,x,y\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "chr-hmm",
		"--in", in,
		"--out", out,
	})
	if err != nil {
		t.Fatalf("run data-extract chr-hmm: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if got != "from,to,tag,extra0,extra1\n100,200,Enh,x,y\n" {
		t.Fatalf("unexpected chr-hmm output:\n%s", got)
	}
}

func TestRunDataExtractWritesTableFile(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw.csv")
	out := filepath.Join(tmp, "gtsa.csv")
	tablePath := filepath.Join(tmp, "gtsa.table.json")
	raw := "t,close\n0,1.1\n1,1.2\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "gtsa",
		"--in", in,
		"--out", out,
		"--value-col", "close",
		"--table-out", tablePath,
		"--table-name", "gtsa_table",
	})
	if err != nil {
		t.Fatalf("run data-extract table-out: %v", err)
	}

	table, err := dataextract.ReadTableFile(tablePath)
	if err != nil {
		t.Fatalf("read table file: %v", err)
	}
	if table.Info.Name != "gtsa_table" {
		t.Fatalf("unexpected table info: %+v", table.Info)
	}
	if len(table.Rows) != 2 {
		t.Fatalf("expected 2 table rows, got %d", len(table.Rows))
	}

	if err := runDataExtract(context.Background(), []string{
		"--table-check", tablePath,
		"--dump-limit", "1",
	}); err != nil {
		t.Fatalf("run data-extract table-check: %v", err)
	}
}

func TestRunDataExtractTableTransformsOnWrite(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw.csv")
	out := filepath.Join(tmp, "gtsa.csv")
	tablePath := filepath.Join(tmp, "gtsa.transforms.json")
	raw := "t,close\n0,0\n1,2\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "gtsa",
		"--in", in,
		"--out", out,
		"--value-col", "close",
		"--table-out", tablePath,
		"--table-scale-asinh",
		"--table-binarize",
		"--table-clean-zero-inputs",
		"--table-stats",
		"--table-zero-counts",
	})
	if err != nil {
		t.Fatalf("run data-extract transforms on write: %v", err)
	}

	table, err := dataextract.ReadTableFile(tablePath)
	if err != nil {
		t.Fatalf("read transformed table: %v", err)
	}
	if len(table.Rows) != 1 {
		t.Fatalf("expected 1 non-zero row after cleaning, got %d", len(table.Rows))
	}
	if got := table.Rows[0].Inputs[0]; got != 1 {
		t.Fatalf("expected binarized value 1, got %f", got)
	}
	if table.Rows[0].Index != 1 {
		t.Fatalf("expected reindexed row, got index=%d", table.Rows[0].Index)
	}
}

func TestRunDataExtractTableCheckScaleMaxAndSave(t *testing.T) {
	tmp := t.TempDir()
	inPath := filepath.Join(tmp, "in.table.json")
	outPath := filepath.Join(tmp, "out.table.json")
	table := dataextract.TableFile{
		Info: dataextract.TableInfo{Name: "scale_test", IVL: 2, OVL: 0, TrnEnd: 2, ValEnd: 2, TstEnd: 2},
		Rows: []dataextract.TableRow{
			{Index: 1, Inputs: []float64{2, 4}},
			{Index: 2, Inputs: []float64{4, 8}},
		},
	}
	if err := dataextract.WriteTableFile(inPath, table); err != nil {
		t.Fatalf("write input table: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--table-check", inPath,
		"--table-scale-max",
		"--table-save", outPath,
	})
	if err != nil {
		t.Fatalf("run data-extract table-check scale-max: %v", err)
	}

	scaled, err := dataextract.ReadTableFile(outPath)
	if err != nil {
		t.Fatalf("read output table: %v", err)
	}
	if got := scaled.Rows[0].Inputs[0]; math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("unexpected scaled value row1 col1: %f", got)
	}
	if got := scaled.Rows[0].Inputs[1]; math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("unexpected scaled value row1 col2: %f", got)
	}
	if got := scaled.Rows[1].Inputs[0]; math.Abs(got-1.0) > 1e-9 {
		t.Fatalf("unexpected scaled value row2 col1: %f", got)
	}
	if got := scaled.Rows[1].Inputs[1]; math.Abs(got-1.0) > 1e-9 {
		t.Fatalf("unexpected scaled value row2 col2: %f", got)
	}
}
