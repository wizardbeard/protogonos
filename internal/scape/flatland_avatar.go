package scape

import "math"

const (
	FlatlandColorRed   = "red"
	FlatlandColorBlue  = "blue"
	FlatlandColorGreen = "green"
	FlatlandColorBlack = "black"
	FlatlandColorBrown = "brown"
	FlatlandColorWhite = "white"
)

const (
	FlatlandStateRespawn    = "respawn"
	FlatlandStateNoRespawn  = "no_respawn"
	FlatlandMetabolicStatic = "static"
)

type FlatlandAvatarStats struct {
	Actuators string
	Sensors   string
	Neurons   int
}

func FlatlandCreatePredatorAvatar(id string, loc FlatlandPoint, initEnergy *float64, stats FlatlandAvatarStats) FlatlandAvatar {
	energy := flatlandDefaultFloat(initEnergy, 1000)
	direction := FlatlandPoint{X: -1 / math.Sqrt2, Y: -1 / math.Sqrt2}
	radius := 6.0
	return FlatlandAvatar{
		Type:      FlatlandObjectPredator,
		ID:        id,
		Specie:    FlatlandObjectPredator,
		Location:  loc,
		Direction: direction,
		Radius:    radius,
		Mass:      6,
		Energy:    energy,
		Actuators: stats.Actuators,
		Sensors:   stats.Sensors,
		Stats:     stats.Neurons,
		Objects: []FlatlandObject{
			flatlandCircle(FlatlandColorRed, loc, radius),
			flatlandDirectionLine(FlatlandColorRed, loc, direction, radius),
		},
	}
}

func FlatlandCreatePreyAvatar(id string, loc FlatlandPoint, initEnergy *float64, stats FlatlandAvatarStats, hasSpear bool) FlatlandAvatar {
	energy := flatlandDefaultFloat(initEnergy, 1000)
	direction := FlatlandPoint{X: 1 / math.Sqrt2, Y: 1 / math.Sqrt2}
	color := FlatlandColorBlue
	objects := []FlatlandObject{flatlandCircle(color, loc, 10)}
	if hasSpear {
		color = FlatlandColorRed
		objects = []FlatlandObject{
			flatlandCircle(color, loc, 10),
			flatlandDirectionLine(color, loc, direction, 10),
		}
	}
	return FlatlandAvatar{
		Type:      FlatlandObjectPrey,
		ID:        id,
		Specie:    FlatlandObjectPrey,
		Location:  loc,
		Direction: direction,
		Radius:    10,
		Mass:      10,
		Energy:    energy,
		Actuators: stats.Actuators,
		Sensors:   stats.Sensors,
		Stats:     stats.Neurons,
		Objects:   objects,
	}
}

func FlatlandCreateAutomatonAvatar(id string, loc FlatlandPoint, angle float64) FlatlandAvatar {
	direction := FlatlandPoint{
		X: (1/math.Sqrt2)*math.Cos(angle) - (1/math.Sqrt2)*math.Sin(angle),
		Y: (1/math.Sqrt2)*math.Sin(angle) + (1/math.Sqrt2)*math.Cos(angle),
	}
	return FlatlandAvatar{
		Type:      FlatlandObjectAutomaton,
		ID:        id,
		Specie:    FlatlandObjectAutomaton,
		Location:  loc,
		Direction: direction,
		Radius:    10,
		Mass:      10,
		Objects:   []FlatlandObject{flatlandCircle(FlatlandColorBlue, loc, 10)},
	}
}

func FlatlandCreatePlantAvatar(id string, loc FlatlandPoint, initEnergy *float64, respawnState, metabolics string) FlatlandAvatar {
	energy := flatlandDefaultFloat(initEnergy, 500)
	mass, radius := flatlandPlantMassRadius(energy, metabolics)
	return FlatlandAvatar{
		Type:      FlatlandObjectPlant,
		ID:        id,
		Specie:    FlatlandObjectPlant,
		Location:  loc,
		Direction: flatlandDefaultDirection(),
		Radius:    radius,
		Mass:      mass,
		Energy:    energy,
		Food:      0,
		Health:    0,
		State:     respawnState,
		Objects:   []FlatlandObject{flatlandCircle(FlatlandColorGreen, loc, radius)},
	}
}

func FlatlandCreatePoisonAvatar(id string, loc FlatlandPoint, initEnergy *float64, respawnState, metabolics string) FlatlandAvatar {
	energy := flatlandDefaultFloat(initEnergy, -2000)
	mass, radius := flatlandPoisonMassRadius(energy, metabolics)
	return FlatlandAvatar{
		Type:      FlatlandObjectPoison,
		ID:        id,
		Specie:    FlatlandObjectPoison,
		Location:  loc,
		Direction: flatlandDefaultDirection(),
		Radius:    radius,
		Mass:      mass,
		Energy:    energy,
		State:     respawnState,
		Objects:   []FlatlandObject{flatlandCircle(FlatlandColorBlack, loc, radius)},
	}
}

func FlatlandCreateRockAvatar(id string, fixture FlatlandCircularFixture) FlatlandAvatar {
	return flatlandCircularFixtureAvatar(FlatlandObjectRock, id, fixture, FlatlandColorBrown)
}

func FlatlandCreateFirePitAvatar(id string, fixture FlatlandCircularFixture) FlatlandAvatar {
	return flatlandCircularFixtureAvatar(FlatlandObjectFirePit, id, fixture, FlatlandColorRed)
}

func FlatlandCreateBeaconAvatar(_ string, fixture FlatlandCircularFixture) FlatlandAvatar {
	return flatlandCircularFixtureAvatar(FlatlandObjectBeacon, FlatlandObjectBeacon, fixture, FlatlandColorWhite)
}

func FlatlandCreateWallAvatar(id string, fixture FlatlandWallFixture) FlatlandAvatar {
	return FlatlandAvatar{
		Type:     FlatlandObjectWall,
		ID:       id,
		Specie:   FlatlandObjectWall,
		Location: FlatlandPoint{X: fixture.X, Y: fixture.Y},
		Energy:   10000,
		State:    fixture.Orientation,
		Objects: []FlatlandObject{{
			Name:   "line",
			Color:  FlatlandColorBrown,
			Pivot:  FlatlandPoint{X: fixture.X, Y: fixture.Y},
			Coords: []FlatlandPoint{{X: fixture.XMin, Y: fixture.YMin}, {X: fixture.XMax, Y: fixture.YMax}},
		}},
	}
}

func flatlandCircularFixtureAvatar(typ, id string, fixture FlatlandCircularFixture, color string) FlatlandAvatar {
	loc := FlatlandPoint{X: fixture.X, Y: fixture.Y}
	return FlatlandAvatar{
		Type:      typ,
		ID:        id,
		Specie:    typ,
		Location:  loc,
		Direction: flatlandDefaultDirection(),
		Radius:    fixture.Radius,
		Energy:    fixture.Energy,
		Objects:   []FlatlandObject{flatlandCircle(color, loc, fixture.Radius)},
	}
}

func flatlandPlantMassRadius(energy float64, metabolics string) (float64, float64) {
	if metabolics == FlatlandMetabolicStatic {
		return 3, 3
	}
	mass := 3 + energy/1000
	return mass, math.Sqrt(mass * 3)
}

func flatlandPoisonMassRadius(energy float64, metabolics string) (float64, float64) {
	if metabolics == FlatlandMetabolicStatic {
		return 3, 3
	}
	mass := 3 + math.Abs(energy)/1000
	return mass, math.Sqrt(mass * 3)
}

func flatlandDefaultFloat(value *float64, fallback float64) float64 {
	if value == nil {
		return fallback
	}
	return *value
}

func flatlandDefaultDirection() FlatlandPoint {
	return FlatlandPoint{X: 1 / math.Sqrt2, Y: 1 / math.Sqrt2}
}

func flatlandCircle(color string, loc FlatlandPoint, radius float64) FlatlandObject {
	return FlatlandObject{
		Name:   "circle",
		Color:  color,
		Pivot:  loc,
		Coords: []FlatlandPoint{loc},
		Radius: radius,
	}
}

func flatlandDirectionLine(color string, loc, direction FlatlandPoint, radius float64) FlatlandObject {
	return FlatlandObject{
		Name:  "line",
		Color: color,
		Pivot: loc,
		Coords: []FlatlandPoint{
			loc,
			{X: loc.X + direction.X*radius*2, Y: loc.Y + direction.Y*radius*2},
		},
	}
}
