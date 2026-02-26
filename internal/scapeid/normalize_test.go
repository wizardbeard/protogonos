package scapeid

import "testing"

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"xor":                     "xor",
		"xor_sim":                 "xor",
		"scape_xor_sim":           "xor",
		"XOR-SIM":                 "xor",
		"pb_sim":                  "pole2-balancing",
		"pb_sim1":                 "pole2-balancing",
		"pole2_balancing":         "pole2-balancing",
		"pole2_balancing_v1":      "pole2-balancing",
		"scape_pb_sim":            "pole2-balancing",
		"dtm_sim":                 "dtm",
		"scape_dtm_sim":           "dtm",
		"flatland_sim":            "flatland",
		"flatlandsim":             "flatland",
		"scape_flatland":          "flatland",
		"scape_flatland_sim":      "flatland",
		"gtsa_sim":                "gtsa",
		"fx_sim":                  "fx",
		"scape_fx_sim":            "fx",
		"scape_GTSA":              "gtsa",
		"scape_LLVMPhaseOrdering": "llvm-phase-ordering",
		"llvm_phase_ordering_sim": "llvm-phase-ordering",
		"epitopes_sim":            "epitopes",
		"scape_epitopes_sim":      "epitopes",
		"regression_mimic":        "regression-mimic",
		"regression_mimic_v1":     "regression-mimic",
		"cart_pole_lite":          "cart-pole-lite",
		"cart_pole_lite_v1":       "cart-pole-lite",
		"epitopes":                "epitopes",
		"epitopes_v1":             "epitopes",
		"custom_sim":              "custom-sim",
		"scape_custom_sim":        "scape-custom-sim",
		"":                        "",
	}

	for in, want := range cases {
		if got := Normalize(in); got != want {
			t.Fatalf("normalize(%q)=%q want=%q", in, got, want)
		}
	}
}
