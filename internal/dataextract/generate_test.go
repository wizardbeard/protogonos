package dataextract

import (
	"path/filepath"
	"testing"
)

func TestGenerateCircuitTestTablesIncludesExpectedShapes(t *testing.T) {
	tables := GenerateCircuitTestTables(7)
	if len(tables) != 8 {
		t.Fatalf("expected 8 generated circuit tables, got %d", len(tables))
	}

	i10o20, ok := tables["i10o20"]
	if !ok {
		t.Fatalf("missing i10o20 table")
	}
	if i10o20.Info.IVL != 10 || i10o20.Info.OVL != 20 {
		t.Fatalf("unexpected i10o20 shape: %+v", i10o20.Info)
	}
	if i10o20.Info.TrnEnd != 500 || i10o20.Info.ValEnd != 600 || i10o20.Info.TstEnd != 700 {
		t.Fatalf("unexpected i10o20 splits: %+v", i10o20.Info)
	}
	if len(i10o20.Rows) != 700 {
		t.Fatalf("unexpected i10o20 row count: %d", len(i10o20.Rows))
	}

	xor, ok := tables["xor_bip"]
	if !ok {
		t.Fatalf("missing xor_bip table")
	}
	if len(xor.Rows) != 4 {
		t.Fatalf("unexpected xor row count: %d", len(xor.Rows))
	}
	if xor.Rows[0].Inputs[0] != -1 || xor.Rows[0].Targets[0] != -1 {
		t.Fatalf("unexpected xor first row: %+v", xor.Rows[0])
	}
	if xor.Rows[3].Inputs[0] != 1 || xor.Rows[3].Inputs[1] != -1 || xor.Rows[3].Targets[0] != 1 {
		t.Fatalf("unexpected xor last row: %+v", xor.Rows[3])
	}
}

func TestGenerateCompetitiveTestTableMetadata(t *testing.T) {
	table := GenerateCompetitiveTestTable(7)
	if table.Info.Name != "i2o0C" {
		t.Fatalf("unexpected table name: %s", table.Info.Name)
	}
	if table.Info.IVL != 2 || table.Info.OVL != 0 {
		t.Fatalf("unexpected shape: %+v", table.Info)
	}
	if table.Info.TrnEnd != 500 || table.Info.ValEnd != 600 || table.Info.TstEnd != 700 {
		t.Fatalf("unexpected split bounds: %+v", table.Info)
	}
	if len(table.Rows) != 700 {
		t.Fatalf("unexpected row count: %d", len(table.Rows))
	}
	if table.Rows[0].Index != 1 || table.Rows[699].Index != 700 {
		t.Fatalf("unexpected row indices first=%d last=%d", table.Rows[0].Index, table.Rows[699].Index)
	}
}

func TestWriteNamedTableFiles(t *testing.T) {
	tmp := t.TempDir()
	tables := map[string]TableFile{
		"a": {Info: TableInfo{Name: "a"}, Rows: []TableRow{{Index: 1}}},
		"b": {Info: TableInfo{Name: "b"}, Rows: []TableRow{{Index: 1}}},
	}
	paths, err := WriteNamedTableFiles(tmp, tables)
	if err != nil {
		t.Fatalf("write named tables: %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("expected two output paths, got %d", len(paths))
	}
	for _, path := range []string{
		filepath.Join(tmp, "a.table.json"),
		filepath.Join(tmp, "b.table.json"),
	} {
		table, err := ReadTableFile(path)
		if err != nil {
			t.Fatalf("read table file %s: %v", path, err)
		}
		if table.Info.Name == "" {
			t.Fatalf("expected non-empty table name in %s", path)
		}
	}
}
