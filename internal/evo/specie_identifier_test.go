package evo

import (
	"testing"
)

func TestTotNSpecieIdentifier(t *testing.T) {
	id := TotNSpecieIdentifier{}
	a := newLinearGenome("a", 1.0)
	b := newLinearGenome("b", 0.5)
	c := newComplexLinearGenome("c", 1.0)

	if id.Identify(a) != id.Identify(b) {
		t.Fatal("expected same tot_n species key")
	}
	if id.Identify(a) == id.Identify(c) {
		t.Fatal("expected different tot_n species key")
	}
}

func TestSpecieIdentifierFromName(t *testing.T) {
	if _, err := SpecieIdentifierFromName("topology"); err != nil {
		t.Fatalf("topology identifier should resolve: %v", err)
	}
	if _, err := SpecieIdentifierFromName("tot_n"); err != nil {
		t.Fatalf("tot_n identifier should resolve: %v", err)
	}
	if _, err := SpecieIdentifierFromName("fingerprint"); err != nil {
		t.Fatalf("fingerprint identifier should resolve: %v", err)
	}
	if _, err := SpecieIdentifierFromName("unknown"); err == nil {
		t.Fatal("expected unknown identifier error")
	}
}

func TestSpecieIdentifierNameFromDistinguishers(t *testing.T) {
	if got := SpecieIdentifierNameFromDistinguishers([]string{"fingerprint"}); got != "fingerprint" {
		t.Fatalf("unexpected identifier from fingerprint: %s", got)
	}
	if got := SpecieIdentifierNameFromDistinguishers([]string{"tot_n"}); got != "tot_n" {
		t.Fatalf("unexpected identifier from tot_n: %s", got)
	}
	if got := SpecieIdentifierNameFromDistinguishers([]string{"pattern"}); got != "topology" {
		t.Fatalf("unexpected identifier from pattern: %s", got)
	}
	if got := SpecieIdentifierNameFromDistinguishers([]string{"unknown"}); got != "" {
		t.Fatalf("unexpected identifier from unknown distinguisher: %s", got)
	}
}
