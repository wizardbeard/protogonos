package dataextract

import (
	"fmt"
	"strings"
)

type TableInfoPatch struct {
	Name   *string
	IVL    *int
	OVL    *int
	TrnEnd *int
	ValEnd *int
	TstEnd *int
	Infer  bool
}

func ApplyTableInfoPatch(table *TableFile, patch TableInfoPatch) error {
	if table == nil {
		return fmt.Errorf("table is required")
	}
	if patch.Infer {
		inferTableInfo(table)
	}
	if patch.Name != nil {
		table.Info.Name = strings.TrimSpace(*patch.Name)
	}
	if patch.IVL != nil {
		table.Info.IVL = *patch.IVL
	}
	if patch.OVL != nil {
		table.Info.OVL = *patch.OVL
	}
	if patch.TrnEnd != nil {
		table.Info.TrnEnd = *patch.TrnEnd
	}
	if patch.ValEnd != nil {
		table.Info.ValEnd = *patch.ValEnd
	}
	if patch.TstEnd != nil {
		table.Info.TstEnd = *patch.TstEnd
	}
	if err := validateTableInfo(*table); err != nil {
		return err
	}
	return nil
}

func inferTableInfo(table *TableFile) {
	if table.Info.Name == "" {
		table.Info.Name = "table"
	}
	if len(table.Rows) == 0 {
		table.Info.TrnEnd = 0
		table.Info.ValEnd = 0
		table.Info.TstEnd = 0
		return
	}
	first := table.Rows[0]
	if table.Info.IVL == 0 {
		table.Info.IVL = len(first.Inputs)
	}
	if table.Info.OVL == 0 && len(first.Targets) > 0 {
		table.Info.OVL = len(first.Targets)
	}
	total := len(table.Rows)
	if table.Info.TrnEnd == 0 {
		table.Info.TrnEnd = total
	}
	if table.Info.TrnEnd > total {
		table.Info.TrnEnd = total
	}
	if table.Info.ValEnd > total {
		table.Info.ValEnd = total
	}
	if table.Info.TstEnd > total {
		table.Info.TstEnd = total
	}
}

func validateTableInfo(table TableFile) error {
	if table.Info.Name == "" {
		return fmt.Errorf("table info name is required")
	}
	if table.Info.IVL < 0 {
		return fmt.Errorf("table info ivl must be >= 0")
	}
	if table.Info.OVL < 0 {
		return fmt.Errorf("table info ovl must be >= 0")
	}
	total := len(table.Rows)
	if table.Info.TrnEnd < 0 || table.Info.TrnEnd > total {
		return fmt.Errorf("table info trn_end out of range: %d (rows=%d)", table.Info.TrnEnd, total)
	}
	if table.Info.ValEnd < 0 || table.Info.ValEnd > total {
		return fmt.Errorf("table info val_end out of range: %d (rows=%d)", table.Info.ValEnd, total)
	}
	if table.Info.TstEnd < 0 || table.Info.TstEnd > total {
		return fmt.Errorf("table info tst_end out of range: %d (rows=%d)", table.Info.TstEnd, total)
	}
	return nil
}
