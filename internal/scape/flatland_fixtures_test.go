package scape

import (
	"math"
	"testing"
)

func TestFlatlandReferenceCircularFixtures(t *testing.T) {
	if seeds := FlatlandCreateSeeds(); len(seeds) != 0 {
		t.Fatalf("expected no reference seeds, got %v", seeds)
	}
	if pillars := FlatlandCreatePillars(); len(pillars) != 0 {
		t.Fatalf("expected no reference pillars, got %v", pillars)
	}
	if water := FlatlandCreateWater(); len(water) != 0 {
		t.Fatalf("expected no reference water fixtures, got %v", water)
	}

	rocks := FlatlandCreateRocks()
	if len(rocks) != 7 {
		t.Fatalf("expected 7 rocks, got %d", len(rocks))
	}
	assertCircularFixture(t, rocks[0], FlatlandObjectRock, 100, 100, 40, math.Inf(1))
	assertCircularFixture(t, rocks[len(rocks)-1], FlatlandObjectRock, 1000, 400, 20, math.Inf(1))

	firePits := FlatlandCreateFirePits()
	if len(firePits) != 7 {
		t.Fatalf("expected 7 fire pits, got %d", len(firePits))
	}
	assertCircularFixture(t, firePits[0], FlatlandObjectFirePit, 600, 100, 50, math.Inf(1))
	assertCircularFixture(t, firePits[len(firePits)-1], FlatlandObjectFirePit, 500, 300, 50, math.Inf(1))

	beacons := FlatlandCreateBeacons()
	if len(beacons) != 1 {
		t.Fatalf("expected 1 beacon, got %d", len(beacons))
	}
	assertCircularFixture(t, beacons[0], FlatlandObjectBeacon, 500, 500, 3, math.Inf(1))
}

func TestFlatlandReferenceWallFixtures(t *testing.T) {
	walls := FlatlandCreateWalls()
	if len(walls) != 6 {
		t.Fatalf("expected 6 walls, got %d", len(walls))
	}

	assertWallFixture(t, walls[0], FlatlandWallX, 150, 300, 100, 200, 300, 300)
	assertWallFixture(t, walls[1], FlatlandWallY, 250, 300, 250, 250, 100, 500)
	assertWallFixture(t, walls[len(walls)-1], FlatlandWallX, 475, 500, 450, 500, 500, 500)
}

func assertCircularFixture(t *testing.T, got FlatlandCircularFixture, typ string, x, y, radius, energy float64) {
	t.Helper()
	if got.Type != typ || got.X != x || got.Y != y || got.Radius != radius || got.Energy != energy {
		t.Fatalf("unexpected circular fixture: got=%+v want={type:%s x:%f y:%f radius:%f energy:%f}", got, typ, x, y, radius, energy)
	}
}

func assertWallFixture(t *testing.T, got FlatlandWallFixture, orientation string, x, y, xMin, xMax, yMin, yMax float64) {
	t.Helper()
	if got.Type != FlatlandObjectWall || got.Orientation != orientation || got.X != x || got.Y != y || got.XMin != xMin || got.XMax != xMax || got.YMin != yMin || got.YMax != yMax || got.Energy != 10000 {
		t.Fatalf("unexpected wall fixture: %+v", got)
	}
}
