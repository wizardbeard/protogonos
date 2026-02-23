package scapeid

import "testing"

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"xor":                     "xor",
		"xor_sim":                 "xor",
		"XOR-SIM":                 "xor",
		"pb_sim":                  "pole2-balancing",
		"pb_sim1":                 "pole2-balancing",
		"pole2_balancing":         "pole2-balancing",
		"dtm_sim":                 "dtm",
		"fx_sim":                  "fx",
		"scape_GTSA":              "gtsa",
		"scape_LLVMPhaseOrdering": "llvm-phase-ordering",
		"regression_mimic":        "regression-mimic",
		"cart_pole_lite":          "cart-pole-lite",
		"epitopes":                "epitopes",
		"":                        "",
	}

	for in, want := range cases {
		if got := Normalize(in); got != want {
			t.Fatalf("normalize(%q)=%q want=%q", in, got, want)
		}
	}
}
