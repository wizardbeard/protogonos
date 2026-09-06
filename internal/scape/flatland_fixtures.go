package scape

import "math"

const (
	FlatlandObjectRock    = "rock"
	FlatlandObjectWall    = "wall"
	FlatlandObjectFirePit = "fire_pit"
	FlatlandObjectBeacon  = "beacon"
)

const (
	FlatlandWallX = "x_wall"
	FlatlandWallY = "y_wall"
)

type FlatlandCircularFixture struct {
	Type   string
	X      float64
	Y      float64
	Radius float64
	Energy float64
}

type FlatlandWallFixture struct {
	Type        string
	Orientation string
	X           float64
	Y           float64
	XMin        float64
	XMax        float64
	YMin        float64
	YMax        float64
	Energy      float64
}

func FlatlandCreateSeeds() []FlatlandCircularFixture {
	return nil
}

func FlatlandCreateRocks() []FlatlandCircularFixture {
	return []FlatlandCircularFixture{
		{Type: FlatlandObjectRock, X: 100, Y: 100, Radius: 40, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 200, Y: 400, Radius: 20, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 300, Y: 500, Radius: 20, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 200, Y: 300, Radius: 60, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 200, Y: 450, Radius: 15, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 300, Y: 100, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectRock, X: 1000, Y: 400, Radius: 20, Energy: math.Inf(1)},
	}
}

func FlatlandCreatePillars() []FlatlandCircularFixture {
	return nil
}

func FlatlandCreateWalls() []FlatlandWallFixture {
	return []FlatlandWallFixture{
		flatlandXWall(300, 100, 200),
		flatlandYWall(250, 100, 500),
		flatlandXWall(200, 400, 450),
		flatlandYWall(400, 200, 300),
		flatlandYWall(500, 400, 500),
		flatlandXWall(500, 450, 500),
	}
}

func FlatlandCreateFirePits() []FlatlandCircularFixture {
	return []FlatlandCircularFixture{
		{Type: FlatlandObjectFirePit, X: 600, Y: 100, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 900, Y: 300, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 800, Y: 200, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 150, Y: 800, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 50, Y: 500, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 600, Y: 800, Radius: 50, Energy: math.Inf(1)},
		{Type: FlatlandObjectFirePit, X: 500, Y: 300, Radius: 50, Energy: math.Inf(1)},
	}
}

func FlatlandCreateWater() []FlatlandCircularFixture {
	return nil
}

func FlatlandCreateBeacons() []FlatlandCircularFixture {
	return []FlatlandCircularFixture{
		{Type: FlatlandObjectBeacon, X: 500, Y: 500, Radius: 3, Energy: math.Inf(1)},
	}
}

func flatlandXWall(y, xMin, xMax float64) FlatlandWallFixture {
	return FlatlandWallFixture{
		Type:        FlatlandObjectWall,
		Orientation: FlatlandWallX,
		X:           (xMin + xMax) / 2,
		Y:           y,
		XMin:        xMin,
		XMax:        xMax,
		YMin:        y,
		YMax:        y,
		Energy:      10000,
	}
}

func flatlandYWall(x, yMin, yMax float64) FlatlandWallFixture {
	return FlatlandWallFixture{
		Type:        FlatlandObjectWall,
		Orientation: FlatlandWallY,
		X:           x,
		Y:           (yMin + yMax) / 2,
		XMin:        x,
		XMax:        x,
		YMin:        yMin,
		YMax:        yMax,
		Energy:      10000,
	}
}
