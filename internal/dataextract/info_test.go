package dataextract

import "testing"

func TestApplyTableInfoPatchInferAndOverride(t *testing.T) {
	table := TableFile{
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{1, 2}, Targets: []float64{0}},
			{Index: 2, Inputs: []float64{3, 4}, Targets: []float64{1}},
		},
	}
	name := "mnist"
	trnEnd := 1
	valEnd := 2
	tstEnd := 2
	if err := ApplyTableInfoPatch(&table, TableInfoPatch{
		Infer:  true,
		Name:   &name,
		TrnEnd: &trnEnd,
		ValEnd: &valEnd,
		TstEnd: &tstEnd,
	}); err != nil {
		t.Fatalf("apply table info patch: %v", err)
	}
	if table.Info.Name != "mnist" {
		t.Fatalf("unexpected name: %+v", table.Info)
	}
	if table.Info.IVL != 2 || table.Info.OVL != 1 {
		t.Fatalf("unexpected inferred shape: %+v", table.Info)
	}
	if table.Info.TrnEnd != 1 || table.Info.ValEnd != 2 || table.Info.TstEnd != 2 {
		t.Fatalf("unexpected split overrides: %+v", table.Info)
	}
}

func TestApplyTableInfoPatchRejectsOutOfRangeSplit(t *testing.T) {
	table := TableFile{
		Info: TableInfo{Name: "bad"},
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{1}},
		},
	}
	tstEnd := 2
	err := ApplyTableInfoPatch(&table, TableInfoPatch{
		Infer:  true,
		TstEnd: &tstEnd,
	})
	if err == nil {
		t.Fatalf("expected out-of-range split error")
	}
}
