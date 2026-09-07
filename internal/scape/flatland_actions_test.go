package scape

import (
	"reflect"
	"testing"
)

func TestFlatlandSpeakAndGestaltOutputMatchReferenceStateUpdates(t *testing.T) {
	avatar := FlatlandAvatar{ID: "a", Energy: 100, Age: 3}

	spoken := FlatlandSpeak(avatar, 0.75)
	if spoken.Sound != 0.75 || spoken.Energy != avatar.Energy || spoken.Age != avatar.Age {
		t.Fatalf("unexpected speak result: %+v", spoken)
	}

	gestalt := []float64{0.1, 0.2}
	updated := FlatlandGestaltOutput(avatar, gestalt)
	if !reflect.DeepEqual(updated.Gestalt, gestalt) {
		t.Fatalf("unexpected gestalt: %+v", updated.Gestalt)
	}
	gestalt[0] = 99
	if updated.Gestalt[0] == 99 {
		t.Fatal("expected gestalt output to copy caller slice")
	}
}

func TestFlatlandCreateOffspringMatchesReferenceGrantAndDenyCosts(t *testing.T) {
	granted := FlatlandCreateOffspring(FlatlandAvatar{ID: "parent", Energy: 1501, Stats: 5}, 1)
	if !granted.OffspringRequested || !granted.OffspringGranted {
		t.Fatalf("expected granted offspring request, got %+v", granted)
	}
	assertClose(t, "granted energy", granted.Energy, 1)
	assertClose(t, "offspring cost", granted.OffspringCost, 1500)
	if granted.OffspringParentID != "parent" {
		t.Fatalf("expected parent id, got %+v", granted)
	}

	denied := FlatlandCreateOffspring(FlatlandAvatar{ID: "parent", Energy: 1500, Stats: 5}, 1)
	if !denied.OffspringRequested || denied.OffspringGranted {
		t.Fatalf("expected denied offspring request, got %+v", denied)
	}
	assertClose(t, "denied energy", denied.Energy, 1450)
	assertClose(t, "denied cost metadata", denied.OffspringCost, 1500)

	noop := FlatlandCreateOffspring(FlatlandAvatar{ID: "parent", Energy: 1500, Stats: 5}, 0)
	if noop.OffspringRequested || noop.OffspringGranted || noop.Energy != 1500 {
		t.Fatalf("expected no-op offspring request, got %+v", noop)
	}
}

func TestFlatlandSpearMatchesReferenceEnergyGate(t *testing.T) {
	high := FlatlandSpear(FlatlandAvatar{Energy: 101}, 1)
	if !high.Spear {
		t.Fatalf("expected high-energy spear enabled: %+v", high)
	}
	assertClose(t, "high energy", high.Energy, 91)

	low := FlatlandSpear(FlatlandAvatar{Energy: 100}, 1)
	if low.Spear {
		t.Fatalf("expected low-energy spear disabled: %+v", low)
	}
	assertClose(t, "low energy", low.Energy, 99)

	disabled := FlatlandSpear(FlatlandAvatar{Energy: 1000, Spear: true}, 0)
	if disabled.Spear || disabled.Energy != 1000 {
		t.Fatalf("expected non-positive spear to disable without cost, got %+v", disabled)
	}
}

func TestFlatlandShootMatchesReferenceEnergyGateAndPreservesSpear(t *testing.T) {
	high := FlatlandShoot(FlatlandAvatar{Energy: 101, Spear: true}, 1)
	assertClose(t, "high energy", high.Energy, 81)
	if !high.Spear {
		t.Fatalf("expected shoot to preserve spear flag: %+v", high)
	}

	low := FlatlandShoot(FlatlandAvatar{Energy: 100}, 1)
	assertClose(t, "low energy", low.Energy, 99)

	noop := FlatlandShoot(FlatlandAvatar{Energy: 100, Spear: true}, 0)
	if noop.Energy != 100 || !noop.Spear {
		t.Fatalf("expected non-positive shoot no-op, got %+v", noop)
	}
}
