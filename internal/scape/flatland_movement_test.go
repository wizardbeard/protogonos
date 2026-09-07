package scape

import (
	"math"
	"testing"
)

func TestFlatlandMoveMatchesReferenceEnergyAndObjectTranslation(t *testing.T) {
	avatar := testFlatlandMovementAvatar("prey")
	moved := FlatlandMove(avatar, 2)

	assertClose(t, "x", moved.Location.X, 12)
	assertClose(t, "y", moved.Location.Y, 20)
	assertClose(t, "energy", moved.Energy, 99.7)
	assertClose(t, "object pivot x", moved.Objects[0].Pivot.X, 12)
	assertClose(t, "object coord x", moved.Objects[0].Coords[0].X, 13)
	if avatar.Location.X != 10 || avatar.Objects[0].Pivot.X != 10 {
		t.Fatalf("expected move to leave input avatar unchanged, got avatar=%+v", avatar)
	}
}

func TestFlatlandMoveUsesReferenceNonPreySpeedScale(t *testing.T) {
	avatar := testFlatlandMovementAvatar("predator")
	moved := FlatlandMove(avatar, 2)

	assertClose(t, "x", moved.Location.X, 11.8)
	assertClose(t, "energy", moved.Energy, 99.72)
}

func TestFlatlandTranslateMatchesReferenceDeltaMove(t *testing.T) {
	avatar := testFlatlandMovementAvatar("prey")
	translated := FlatlandTranslate(avatar, FlatlandPoint{X: 3, Y: 4})

	assertClose(t, "x", translated.Location.X, 13)
	assertClose(t, "y", translated.Location.Y, 24)
	assertClose(t, "energy", translated.Energy, 99.4)
	assertClose(t, "object coord y", translated.Objects[0].Coords[1].Y, 25)
}

func TestFlatlandRotateMatchesReferenceQuarterPiRatio(t *testing.T) {
	avatar := testFlatlandMovementAvatar("prey")
	rotated := FlatlandRotate(avatar, 1)

	assertClose(t, "direction x", rotated.Direction.X, math.Sqrt(0.5))
	assertClose(t, "direction y", rotated.Direction.Y, math.Sqrt(0.5))
	assertClose(t, "energy", rotated.Energy, 100-0.1*math.Pi/4-0.1)
	assertClose(t, "rotated coord x", rotated.Objects[0].Coords[0].X, 10+math.Sqrt(0.5))
	assertClose(t, "rotated coord y", rotated.Objects[0].Coords[0].Y, 20+math.Sqrt(0.5))
}

func TestFlatlandTwoWheelsMatchesReferenceRotateThenMove(t *testing.T) {
	avatar := testFlatlandMovementAvatar("prey")
	got := FlatlandTwoWheels(avatar, 1, -1)
	want := FlatlandMove(FlatlandRotate(avatar, 2), 0)

	if got.Age != avatar.Age+1 {
		t.Fatalf("expected age increment, got %d", got.Age)
	}
	assertClose(t, "x", got.Location.X, want.Location.X)
	assertClose(t, "y", got.Location.Y, want.Location.Y)
	assertClose(t, "energy", got.Energy, want.Energy)

	speed, angle := FlatlandTwoWheelToMoveRotate(1, -1)
	assertClose(t, "speed", speed, 0)
	assertClose(t, "angle", angle, 2)
}

func testFlatlandMovementAvatar(typ string) FlatlandAvatar {
	return FlatlandAvatar{
		Type:      typ,
		Location:  FlatlandPoint{X: 10, Y: 20},
		Direction: FlatlandPoint{X: 1, Y: 0},
		Energy:    100,
		Age:       7,
		Objects: []FlatlandObject{
			{
				Name:  "line",
				ID:    "o1",
				Color: "blue",
				Pivot: FlatlandPoint{X: 10, Y: 20},
				Coords: []FlatlandPoint{
					{X: 11, Y: 20},
					{X: 10, Y: 21},
				},
			},
		},
	}
}

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s: got %0.15f want %0.15f", name, got, want)
	}
}
