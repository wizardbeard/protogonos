package scape

import (
	"math"
	"testing"
)

func TestFlatlandCreatePredatorAvatarMatchesReferenceShape(t *testing.T) {
	energy := 1200.0
	avatar := FlatlandCreatePredatorAvatar("pred", FlatlandPoint{X: 300, Y: 400}, &energy, FlatlandAvatarStats{
		Actuators: "cf",
		Sensors:   "ct",
		Neurons:   7,
	})

	if avatar.ID != "pred" || avatar.Type != FlatlandObjectPredator || avatar.Specie != FlatlandObjectPredator {
		t.Fatalf("unexpected predator identity: %+v", avatar)
	}
	assertClose(t, "direction x", avatar.Direction.X, -1/math.Sqrt2)
	assertClose(t, "direction y", avatar.Direction.Y, -1/math.Sqrt2)
	assertClose(t, "energy", avatar.Energy, 1200)
	assertClose(t, "mass", avatar.Mass, 6)
	assertClose(t, "radius", avatar.Radius, 6)
	if avatar.Actuators != "cf" || avatar.Sensors != "ct" || avatar.Stats != 7 {
		t.Fatalf("unexpected stats fields: %+v", avatar)
	}
	assertFlatlandObject(t, avatar.Objects[0], "circle", FlatlandColorRed, 300, 400, 6, 1)
	assertFlatlandObject(t, avatar.Objects[1], "line", FlatlandColorRed, 300, 400, 0, 2)
}

func TestFlatlandCreatePreyAvatarDefaultsAndSpearShape(t *testing.T) {
	avatar := FlatlandCreatePreyAvatar("prey", FlatlandPoint{X: 10, Y: 20}, nil, FlatlandAvatarStats{}, false)
	assertClose(t, "default prey energy", avatar.Energy, 1000)
	assertClose(t, "prey radius", avatar.Radius, 10)
	assertClose(t, "prey mass", avatar.Mass, 10)
	if len(avatar.Objects) != 1 || avatar.Objects[0].Color != FlatlandColorBlue {
		t.Fatalf("expected non-spear prey blue circle, got %+v", avatar.Objects)
	}

	spear := FlatlandCreatePreyAvatar("prey", FlatlandPoint{X: 10, Y: 20}, nil, FlatlandAvatarStats{}, true)
	if len(spear.Objects) != 2 || spear.Objects[0].Color != FlatlandColorRed || spear.Objects[1].Name != "line" {
		t.Fatalf("expected spear prey red circle+line, got %+v", spear.Objects)
	}
}

func TestFlatlandCreateAutomatonAvatarMatchesReferenceShape(t *testing.T) {
	avatar := FlatlandCreateAutomatonAvatar("auto", FlatlandPoint{X: 400, Y: 200}, 0)
	if avatar.Type != FlatlandObjectAutomaton || avatar.ID != "auto" {
		t.Fatalf("unexpected automaton identity: %+v", avatar)
	}
	assertClose(t, "direction x", avatar.Direction.X, 1/math.Sqrt2)
	assertClose(t, "direction y", avatar.Direction.Y, 1/math.Sqrt2)
	assertClose(t, "mass", avatar.Mass, 10)
	assertClose(t, "radius", avatar.Radius, 10)
	assertFlatlandObject(t, avatar.Objects[0], "circle", FlatlandColorBlue, 400, 200, 10, 1)
}

func TestFlatlandCreatePlantAndPoisonAvatarsMatchDefaults(t *testing.T) {
	plant := FlatlandCreatePlantAvatar("plant", FlatlandPoint{X: 1, Y: 2}, nil, FlatlandStateRespawn, FlatlandMetabolicStatic)
	assertClose(t, "plant energy", plant.Energy, 500)
	assertClose(t, "plant mass", plant.Mass, 3)
	assertClose(t, "plant radius", plant.Radius, 3)
	if plant.State != FlatlandStateRespawn || plant.Food != 0 || plant.Health != 0 {
		t.Fatalf("unexpected plant state: %+v", plant)
	}
	assertFlatlandObject(t, plant.Objects[0], "circle", FlatlandColorGreen, 1, 2, 3, 1)

	poison := FlatlandCreatePoisonAvatar("poison", FlatlandPoint{X: 3, Y: 4}, nil, FlatlandStateNoRespawn, FlatlandMetabolicStatic)
	assertClose(t, "poison energy", poison.Energy, -2000)
	assertClose(t, "poison mass", poison.Mass, 3)
	assertClose(t, "poison radius", poison.Radius, 3)
	if poison.State != FlatlandStateNoRespawn {
		t.Fatalf("unexpected poison state: %+v", poison)
	}
	assertFlatlandObject(t, poison.Objects[0], "circle", FlatlandColorBlack, 3, 4, 3, 1)
}

func TestFlatlandCreateStaticObjectAvatarsFromFixtures(t *testing.T) {
	rock := FlatlandCreateRockAvatar("rock-1", FlatlandCreateRocks()[0])
	if rock.Type != FlatlandObjectRock || rock.ID != "rock-1" {
		t.Fatalf("unexpected rock: %+v", rock)
	}
	assertClose(t, "rock radius", rock.Radius, 40)
	assertFlatlandObject(t, rock.Objects[0], "circle", FlatlandColorBrown, 100, 100, 40, 1)

	firePit := FlatlandCreateFirePitAvatar("fire-1", FlatlandCreateFirePits()[0])
	if firePit.Type != FlatlandObjectFirePit || firePit.ID != "fire-1" {
		t.Fatalf("unexpected fire pit: %+v", firePit)
	}
	assertFlatlandObject(t, firePit.Objects[0], "circle", FlatlandColorRed, 600, 100, 50, 1)

	beacon := FlatlandCreateBeaconAvatar("ignored", FlatlandCreateBeacons()[0])
	if beacon.Type != FlatlandObjectBeacon || beacon.ID != FlatlandObjectBeacon {
		t.Fatalf("expected reference beacon ID behavior, got %+v", beacon)
	}
	assertFlatlandObject(t, beacon.Objects[0], "circle", FlatlandColorWhite, 500, 500, 3, 1)
}

func TestFlatlandCreateWallAvatarMatchesReferenceLineObject(t *testing.T) {
	wall := FlatlandCreateWallAvatar("wall-1", FlatlandCreateWalls()[0])
	if wall.Type != FlatlandObjectWall || wall.ID != "wall-1" || wall.State != FlatlandWallX {
		t.Fatalf("unexpected wall identity: %+v", wall)
	}
	assertClose(t, "wall energy", wall.Energy, 10000)
	assertFlatlandObject(t, wall.Objects[0], "line", FlatlandColorBrown, 150, 300, 0, 2)
	assertClose(t, "line start x", wall.Objects[0].Coords[0].X, 100)
	assertClose(t, "line end x", wall.Objects[0].Coords[1].X, 200)
}

func assertFlatlandObject(t *testing.T, got FlatlandObject, name, color string, pivotX, pivotY, radius float64, coordCount int) {
	t.Helper()
	if got.Name != name || got.Color != color || len(got.Coords) != coordCount {
		t.Fatalf("unexpected object identity: %+v", got)
	}
	assertClose(t, "object pivot x", got.Pivot.X, pivotX)
	assertClose(t, "object pivot y", got.Pivot.Y, pivotY)
	assertClose(t, "object radius", got.Radius, radius)
}
