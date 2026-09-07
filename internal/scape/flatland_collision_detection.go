package scape

import "math"

const FlatlandMaxEnergy = 10000

type FlatlandCollisionDetectionResult struct {
	Operator    FlatlandAvatar
	Avatars     []FlatlandAvatar
	EnergyDelta float64
	Kills       int
}

func FlatlandCollisionDetection(operator FlatlandAvatar, avatars []FlatlandAvatar) FlatlandCollisionDetectionResult {
	current := operator
	out := make([]FlatlandAvatar, 0, len(avatars))
	energyAcc := 0.0
	killsAcc := 0

	for _, target := range avatars {
		if target.ID == current.ID {
			out = append(out, target)
			continue
		}
		if target.Type == FlatlandObjectWall {
			if wall, ok := FlatlandWallFixtureFromAvatar(target); ok {
				current = FlatlandWorldWallCollision(current, wall)
			}
			out = append(out, target)
			continue
		}

		collision := flatlandAvatarCollision(current, target)
		penetration := flatlandAvatarPenetration(current, target)
		behavior := FlatlandWorldBehavior(false, false, current, target)
		if collision || penetration {
			behavior = FlatlandWorldBehavior(collision, penetration, current, target)
		}
		current = behavior.Operator
		energyAcc += behavior.EnergyDelta

		switch behavior.Order {
		case FlatlandWorldOrderDestroy:
			killsAcc++
		case FlatlandWorldOrderPlantEaten:
			if behavior.EnergyDelta > 0 {
				killsAcc++
			}
			if behavior.Target.State == FlatlandStateRespawn {
				out = append(out, FlatlandRespawnAvatarAtCurrentLocation(behavior.Target))
			}
		case FlatlandWorldOrderPoisonEaten:
			if behavior.Target.State == FlatlandStateRespawn {
				out = append(out, FlatlandRespawnAvatarAtCurrentLocation(behavior.Target))
			}
		default:
			out = append(out, behavior.Target)
		}
	}

	if energyAcc != 0 {
		current.Energy = clamp(current.Energy+energyAcc, -FlatlandMaxEnergy, FlatlandMaxEnergy)
		current.Kills += killsAcc
	}
	out = flatlandReplaceOrAppendAvatar(out, current)
	return FlatlandCollisionDetectionResult{
		Operator:    current,
		Avatars:     out,
		EnergyDelta: energyAcc,
		Kills:       killsAcc,
	}
}

func FlatlandRespawnAvatarAtCurrentLocation(avatar FlatlandAvatar) FlatlandAvatar {
	switch avatar.Type {
	case FlatlandObjectPlant:
		energy := 500.0
		return FlatlandCreatePlantAvatar(avatar.ID, avatar.Location, &energy, avatar.State, FlatlandMetabolicStatic)
	case FlatlandObjectPoison:
		energy := -2000.0
		return FlatlandCreatePoisonAvatar(avatar.ID, avatar.Location, &energy, avatar.State, FlatlandMetabolicStatic)
	default:
		return avatar
	}
}

func FlatlandWallFixtureFromAvatar(avatar FlatlandAvatar) (FlatlandWallFixture, bool) {
	if avatar.Type != FlatlandObjectWall || len(avatar.Objects) == 0 || len(avatar.Objects[0].Coords) < 2 {
		return FlatlandWallFixture{}, false
	}
	a := avatar.Objects[0].Coords[0]
	b := avatar.Objects[0].Coords[1]
	orientation := avatar.State
	if orientation == "" {
		if a.Y == b.Y {
			orientation = FlatlandWallX
		} else {
			orientation = FlatlandWallY
		}
	}
	return FlatlandWallFixture{
		Type:        FlatlandObjectWall,
		Orientation: orientation,
		X:           avatar.Location.X,
		Y:           avatar.Location.Y,
		XMin:        math.Min(a.X, b.X),
		XMax:        math.Max(a.X, b.X),
		YMin:        math.Min(a.Y, b.Y),
		YMax:        math.Max(a.Y, b.Y),
		Energy:      avatar.Energy,
	}, true
}

func flatlandAvatarCollision(operator, target FlatlandAvatar) bool {
	return math.Hypot(operator.Location.X-target.Location.X, operator.Location.Y-target.Location.Y) < operator.Radius+target.Radius
}

func flatlandAvatarPenetration(operator, target FlatlandAvatar) bool {
	if operator.Type != FlatlandObjectPredator && !operator.Spear {
		return false
	}
	dirLen := math.Hypot(operator.Direction.X, operator.Direction.Y)
	if dirLen == 0 {
		return false
	}
	dx := target.Location.X - operator.Location.X
	dy := target.Location.Y - operator.Location.Y
	unitX := operator.Direction.X / dirLen
	unitY := operator.Direction.Y / dirLen
	projection := dx*unitX + dy*unitY
	if projection < 0 || projection >= 2+operator.Radius {
		return false
	}
	crossTrack := math.Abs(dx*unitY - dy*unitX)
	return crossTrack <= target.Radius
}

func flatlandReplaceOrAppendAvatar(avatars []FlatlandAvatar, avatar FlatlandAvatar) []FlatlandAvatar {
	for i := range avatars {
		if avatars[i].ID == avatar.ID {
			avatars[i] = avatar
			return avatars
		}
	}
	return append(avatars, avatar)
}
