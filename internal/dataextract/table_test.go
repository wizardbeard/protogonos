package dataextract

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildTableFromExtractedCSVForEpitopes(t *testing.T) {
	in := strings.NewReader("signal,memory,class,seq0,seq1\n0.1,0,1,4,10\n-0.2,0.1,0,2,2\n")
	table, err := BuildTableFromExtractedCSV(in, BuildTableOptions{
		Scape: "epitopes",
		Name:  "epitopes_test",
	})
	if err != nil {
		t.Fatalf("build table: %v", err)
	}
	if table.Info.Name != "epitopes_test" {
		t.Fatalf("unexpected table name: %+v", table.Info)
	}
	if table.Info.IVL != 4 || table.Info.OVL != 1 {
		t.Fatalf("unexpected ivl/ovl: %+v", table.Info)
	}
	if len(table.Rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(table.Rows))
	}
	if len(table.Rows[0].Inputs) != 4 || len(table.Rows[0].Targets) != 1 {
		t.Fatalf("unexpected row shape: %+v", table.Rows[0])
	}
}

func TestBuildTableFromExtractedCSVForChrHMM(t *testing.T) {
	in := strings.NewReader("from,to,tag,extra0\n100,200,Enh,x\n")
	table, err := BuildTableFromExtractedCSV(in, BuildTableOptions{
		Scape: "chr-hmm",
		Name:  "chr_hmm_test",
	})
	if err != nil {
		t.Fatalf("build chr_hmm table: %v", err)
	}
	if len(table.Rows) != 1 {
		t.Fatalf("expected one row, got %d", len(table.Rows))
	}
	if len(table.Rows[0].Fields) != 4 {
		t.Fatalf("expected fields row for chr_hmm, got %+v", table.Rows[0])
	}
}

func TestWriteReadAndDumpTableFile(t *testing.T) {
	table := TableFile{
		Info: TableInfo{
			Name:   "test_table",
			IVL:    2,
			OVL:    1,
			TrnEnd: 3,
			ValEnd: 3,
			TstEnd: 3,
		},
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{1, 2}, Targets: []float64{1}},
			{Index: 2, Inputs: []float64{3, 4}, Targets: []float64{0}},
		},
	}
	path := filepath.Join(t.TempDir(), "table.json")
	if err := WriteTableFile(path, table); err != nil {
		t.Fatalf("write table file: %v", err)
	}
	loaded, err := ReadTableFile(path)
	if err != nil {
		t.Fatalf("read table file: %v", err)
	}
	if loaded.Info.Name != "test_table" || len(loaded.Rows) != 2 {
		t.Fatalf("unexpected loaded table: %+v", loaded)
	}
	dump := DumpTable(loaded, 1)
	if len(dump) != 1 || dump[0].Index != 1 {
		t.Fatalf("unexpected table dump: %+v", dump)
	}
}
