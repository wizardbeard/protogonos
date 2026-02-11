package evo

import "testing"

func TestComputeGenomeSignatureDeterministic(t *testing.T) {
	g := newComplexLinearGenome("g", 1.0)
	s1 := ComputeGenomeSignature(g)
	s2 := ComputeGenomeSignature(g)
	if s1.Fingerprint == "" {
		t.Fatal("expected non-empty fingerprint")
	}
	if s1.Fingerprint != s2.Fingerprint {
		t.Fatalf("expected deterministic fingerprint: %s != %s", s1.Fingerprint, s2.Fingerprint)
	}
}

func TestComputeGenomeSignatureChangesWithTopology(t *testing.T) {
	a := newLinearGenome("a", 1.0)
	b := newComplexLinearGenome("b", 1.0)
	sa := ComputeGenomeSignature(a)
	sb := ComputeGenomeSignature(b)
	if sa.Fingerprint == sb.Fingerprint {
		t.Fatalf("expected different fingerprints for different topology: %s", sa.Fingerprint)
	}
}
