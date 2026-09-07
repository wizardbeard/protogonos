package scape

import "math"

const FlatlandMaxEnergy = 10000

const (
	FlatlandRespawnMaxX = 800
	FlatlandRespawnMaxY = 500
)

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

	for i, target := range avatars {
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
				out = append(out, FlatlandRespawnAvatar(flatlandRespawnContext(out, target, avatars[i+1:]), behavior.Target))
			}
		case FlatlandWorldOrderPoisonEaten:
			if behavior.Target.State == FlatlandStateRespawn {
				out = append(out, FlatlandRespawnAvatar(flatlandRespawnContext(out, target, avatars[i+1:]), behavior.Target))
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

func FlatlandRespawnAvatar(obstacles []FlatlandAvatar, avatar FlatlandAvatar) FlatlandAvatar {
	loc := FlatlandReturnValid(obstacles)
	return FlatlandRespawnAvatarAt(avatar, loc)
}

func FlatlandRespawnAvatarWithCandidates(obstacles []FlatlandAvatar, avatar FlatlandAvatar, candidates []FlatlandPoint) FlatlandAvatar {
	loc := FlatlandReturnValidFromCandidates(obstacles, candidates)
	return FlatlandRespawnAvatarAt(avatar, loc)
}

func FlatlandRespawnAvatarAt(avatar FlatlandAvatar, loc FlatlandPoint) FlatlandAvatar {
	avatar.Location = loc
	switch avatar.Type {
	case FlatlandObjectPlant:
		avatar.Energy = 500
		avatar.Objects = flatlandRespawnObjects(avatar.Objects, FlatlandColorGreen, loc)
		return avatar
	case FlatlandObjectPoison:
		avatar.Energy = -2000
		avatar.Objects = flatlandRespawnObjects(avatar.Objects, FlatlandColorBlack, loc)
		return avatar
	default:
		return avatar
	}
}

func FlatlandReturnValid(avatars []FlatlandAvatar) FlatlandPoint {
	obstacles := flatlandRespawnObstacles(avatars)
	for y := 1; y <= FlatlandRespawnMaxY; y++ {
		for x := 1; x <= FlatlandRespawnMaxX; x++ {
			loc := FlatlandPoint{X: float64(x), Y: float64(y)}
			if flatlandValidRespawnLocation(obstacles, loc) {
				return loc
			}
		}
	}
	return FlatlandPoint{X: 1, Y: 1}
}

func FlatlandReturnValidFromCandidates(avatars []FlatlandAvatar, candidates []FlatlandPoint) FlatlandPoint {
	obstacles := flatlandRespawnObstacles(avatars)
	for _, candidate := range candidates {
		if flatlandValidRespawnLocation(obstacles, candidate) {
			return candidate
		}
	}
	return FlatlandPoint{X: 1, Y: 1}
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

func flatlandRespawnObstacles(avatars []FlatlandAvatar) []FlatlandAvatar {
	obstacles := make([]FlatlandAvatar, 0, len(avatars))
	for _, avatar := range avatars {
		if avatar.Type == FlatlandObjectRock || avatar.Type == FlatlandObjectFirePit {
			obstacles = append(obstacles, avatar)
		}
	}
	return obstacles
}

func flatlandRespawnContext(prefix []FlatlandAvatar, target FlatlandAvatar, suffix []FlatlandAvatar) []FlatlandAvatar {
	context := make([]FlatlandAvatar, 0, len(prefix)+1+len(suffix))
	context = append(context, target)
	context = append(context, suffix...)
	context = append(context, prefix...)
	return context
}

func flatlandRespawnObjects(objects []FlatlandObject, color string, loc FlatlandPoint) []FlatlandObject {
	out := make([]FlatlandObject, len(objects))
	for i, object := range objects {
		out[i] = object
		out[i].Color = color
		out[i].Pivot = loc
		out[i].Coords = make([]FlatlandPoint, len(object.Coords))
		for j := range object.Coords {
			out[i].Coords[j] = loc
		}
	}
	return out
}

func flatlandValidRespawnLocation(obstacles []FlatlandAvatar, loc FlatlandPoint) bool {
	for _, obstacle := range obstacles {
		distance := math.Hypot(loc.X-obstacle.Location.X, loc.Y-obstacle.Location.Y)
		if distance < obstacle.Radius+2 {
			return false
		}
	}
	return true
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
