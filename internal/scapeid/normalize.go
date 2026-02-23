package scapeid

import "strings"

// Normalize canonicalizes scape names and reference aliases.
func Normalize(name string) string {
	normalized := strings.TrimSpace(strings.ToLower(name))
	normalized = strings.ReplaceAll(normalized, "_", "-")
	normalized = strings.ReplaceAll(normalized, " ", "-")
	normalized = strings.Trim(normalized, "-")

	switch normalized {
	case "":
		return ""
	case "xor", "xor-sim", "xorsim":
		return "xor"
	case "regression-mimic", "regressionmimic", "regression-mimic-v1":
		return "regression-mimic"
	case "cart-pole-lite", "cartpolelite", "cart-pole-lite-v1":
		return "cart-pole-lite"
	case "pole2-balancing", "pole2balancing", "pole2-balancing-v1", "pb-sim", "pb-sim1", "pbsim", "pbsim1", "pole2-balancing-sim":
		return "pole2-balancing"
	case "dtm", "dtm-sim", "dtmsim":
		return "dtm"
	case "flatland":
		return "flatland"
	case "gtsa", "scape-gtsa", "scapegtsa":
		return "gtsa"
	case "fx", "fx-sim", "fxsim":
		return "fx"
	case "epitopes":
		return "epitopes"
	case "llvm-phase-ordering", "llvmphaseordering", "scape-llvmphaseordering", "scape-llvm-phase-ordering":
		return "llvm-phase-ordering"
	default:
		return normalized
	}
}
