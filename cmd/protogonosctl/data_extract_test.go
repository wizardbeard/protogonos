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

func TestRunDataExtractSimple(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw_simple.csv")
	out := filepath.Join(tmp, "simple.csv")
	raw := "x,y\n1,2\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "simple",
		"--in", in,
		"--out", out,
	})
	if err != nil {
		t.Fatalf("run data-extract simple: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if got := string(data); got != "x,y\n1,2\n" {
		t.Fatalf("unexpected simple output:\n%s", got)
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

func TestRunDataExtractChromHMMExpanded(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "chrom_hmm_raw.csv")
	out := filepath.Join(tmp, "chrom_hmm_expanded.csv")
	raw := "chr,from,to,tag\nchr22,100,500,Enh\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "chrom-hmm-expanded",
		"--in", in,
		"--out", out,
		"--chrom-step", "200",
	})
	if err != nil {
		t.Fatalf("run data-extract chrom-hmm-expanded: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	got := string(data)
	if !strings.HasPrefix(got, "tag_Enh,bp_index,tag,chrom\n") {
		t.Fatalf("unexpected expanded header:\n%s", got)
	}
	if !strings.Contains(got, "1,100,Enh,chr22\n") {
		t.Fatalf("expected expanded first row:\n%s", got)
	}
}

func TestRunDataExtractMinesVsRocks(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "rw_raw.csv")
	out := filepath.Join(tmp, "rw.csv")
	raw := "*CM001,0.1,0.2\nCR001,0.3,0.4\n"
	if err := os.WriteFile(in, []byte(raw), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--scape", "mines-vs-rocks",
		"--in", in,
		"--out", out,
		"--has-header=false",
	})
	if err != nil {
		t.Fatalf("run data-extract mines-vs-rocks: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if got := string(data); got != "split,feature0,feature1,class0,class1\n0,0.1,0.2,1,0\n1,0.3,0.4,0,1\n" {
		t.Fatalf("unexpected mines-vs-rocks output:\n%s", got)
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

func TestRunDataExtractTableCheckResolutionAndSave(t *testing.T) {
	tmp := t.TempDir()
	inPath := filepath.Join(tmp, "in.table.json")
	outPath := filepath.Join(tmp, "out.table.json")
	table := dataextract.TableFile{
		Info: dataextract.TableInfo{Name: "resolution_test", IVL: 1, OVL: 0, TrnEnd: 4, ValEnd: 4, TstEnd: 4},
		Rows: []dataextract.TableRow{
			{Index: 1, Inputs: []float64{0}},
			{Index: 2, Inputs: []float64{0}},
			{Index: 3, Inputs: []float64{1}},
			{Index: 4, Inputs: []float64{3}},
		},
	}
	if err := dataextract.WriteTableFile(inPath, table); err != nil {
		t.Fatalf("write input table: %v", err)
	}

	err := runDataExtract(context.Background(), []string{
		"--table-check", inPath,
		"--table-resolution", "2",
		"--table-resolution-drop-zero-run", "1",
		"--table-resolution-asinh=true",
		"--table-save", outPath,
	})
	if err != nil {
		t.Fatalf("run data-extract table-check resolution: %v", err)
	}

	resolved, err := dataextract.ReadTableFile(outPath)
	if err != nil {
		t.Fatalf("read output table: %v", err)
	}
	if len(resolved.Rows) != 1 {
		t.Fatalf("expected one resolved row, got %d", len(resolved.Rows))
	}
	if got := resolved.Rows[0].Inputs[0]; math.Abs(got-math.Asinh(2)) > 1e-9 {
		t.Fatalf("unexpected resolved value: %f", got)
	}
}

func TestRunDataExtractGenerateCircuitTables(t *testing.T) {
	tmp := t.TempDir()
	err := runDataExtract(context.Background(), []string{
		"--generate-circuit-tests", tmp,
		"--seed", "7",
	})
	if err != nil {
		t.Fatalf("run data-extract generate-circuit-tests: %v", err)
	}

	i10o20Path := filepath.Join(tmp, "i10o20.table.json")
	i10o20, err := dataextract.ReadTableFile(i10o20Path)
	if err != nil {
		t.Fatalf("read generated i10o20 table: %v", err)
	}
	if i10o20.Info.IVL != 10 || i10o20.Info.OVL != 20 {
		t.Fatalf("unexpected i10o20 shape: %+v", i10o20.Info)
	}
	if len(i10o20.Rows) != 700 {
		t.Fatalf("expected i10o20 total rows=700, got %d", len(i10o20.Rows))
	}

	xorPath := filepath.Join(tmp, "xor_bip.table.json")
	xor, err := dataextract.ReadTableFile(xorPath)
	if err != nil {
		t.Fatalf("read generated xor_bip table: %v", err)
	}
	if len(xor.Rows) != 4 {
		t.Fatalf("expected xor rows=4, got %d", len(xor.Rows))
	}
}

func TestRunDataExtractGenerateCompetitiveTable(t *testing.T) {
	tmp := t.TempDir()
	err := runDataExtract(context.Background(), []string{
		"--generate-competitive-tests", tmp,
		"--seed", "11",
	})
	if err != nil {
		t.Fatalf("run data-extract generate-competitive-tests: %v", err)
	}

	path := filepath.Join(tmp, "i2o0C.table.json")
	table, err := dataextract.ReadTableFile(path)
	if err != nil {
		t.Fatalf("read generated competitive table: %v", err)
	}
	if table.Info.Name != "i2o0C" {
		t.Fatalf("unexpected table info: %+v", table.Info)
	}
	if table.Info.TrnEnd != 500 || table.Info.ValEnd != 600 || table.Info.TstEnd != 700 {
		t.Fatalf("unexpected split bounds: %+v", table.Info)
	}
	if len(table.Rows) != 700 {
		t.Fatalf("expected total rows=700, got %d", len(table.Rows))
	}
}

func TestRunDataExtractTableInfoOverridesOnWrite(t *testing.T) {
	tmp := t.TempDir()
	in := filepath.Join(tmp, "raw.csv")
	out := filepath.Join(tmp, "gtsa.csv")
	tablePath := filepath.Join(tmp, "custom.table.json")
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
		"--table-info-name", "mnist",
		"--table-info-trn-end", "1",
		"--table-info-val-end", "2",
		"--table-info-tst-end", "2",
		"--table-info-ivl", "1",
	})
	if err != nil {
		t.Fatalf("run data-extract info overrides on write: %v", err)
	}

	table, err := dataextract.ReadTableFile(tablePath)
	if err != nil {
		t.Fatalf("read output table: %v", err)
	}
	if table.Info.Name != "mnist" {
		t.Fatalf("unexpected table name: %+v", table.Info)
	}
	if table.Info.IVL != 1 || table.Info.TrnEnd != 1 || table.Info.ValEnd != 2 || table.Info.TstEnd != 2 {
		t.Fatalf("unexpected table info: %+v", table.Info)
	}
}

func TestRunDataExtractTableCheckInferInfoAndSave(t *testing.T) {
	tmp := t.TempDir()
	inPath := filepath.Join(tmp, "in.table.json")
	outPath := filepath.Join(tmp, "out.table.json")
	table := dataextract.TableFile{
		Info: dataextract.TableInfo{},
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
		"--table-save", outPath,
	})
	if err != nil {
		t.Fatalf("run data-extract table-check infer: %v", err)
	}

	updated, err := dataextract.ReadTableFile(outPath)
	if err != nil {
		t.Fatalf("read updated table: %v", err)
	}
	if updated.Info.Name != "table" {
		t.Fatalf("expected inferred table name, got %+v", updated.Info)
	}
	if updated.Info.IVL != 2 || updated.Info.TrnEnd != 2 {
		t.Fatalf("unexpected inferred table info: %+v", updated.Info)
	}
}
