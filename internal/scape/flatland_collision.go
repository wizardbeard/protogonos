package scape

import "math"

func FlatlandPush(pusher, avatar FlatlandAvatar, strength float64) FlatlandAvatar {
	if pusher.Energy <= avatar.Energy {
		return avatar
	}

	dx := avatar.Location.X - pusher.Location.X
	dy := avatar.Location.Y - pusher.Location.Y
	distance := math.Hypot(dx, dy)
	minDistance := pusher.Radius + avatar.Radius

	minPushX := 0.0
	minPushY := 0.0
	if distance != 0 {
		minPushX = (minDistance/distance)*dx - dx
		minPushY = (minDistance/distance)*dy - dy
	}

	pushX := minPushX
	if dx != 0 {
		pushX += math.Copysign(strength, dx)
	}
	pushY := minPushY
	if dy != 0 {
		pushY += math.Copysign(strength, dy)
	}

	avatar.Location.X += pushX
	avatar.Location.Y += pushY
	avatar.Energy -= 10 * strength
	avatar.Objects = flatlandTranslateObjects(avatar.Objects, pushX, pushY)
	return avatar
}

func FlatlandResist(origin FlatlandPoint, avatar FlatlandAvatar) FlatlandAvatar {
	dx := avatar.Location.X - origin.X
	dy := avatar.Location.Y - origin.Y
	distance := math.Hypot(dx, dy)
	if distance == 0 {
		return avatar
	}

	pushX := (avatar.Radius/distance)*dx - dx
	pushY := (avatar.Radius/distance)*dy - dy
	avatar.Location.X += pushX
	avatar.Location.Y += pushY
	avatar.Objects = flatlandTranslateObjects(avatar.Objects, pushX, pushY)
	return avatar
}

func FlatlandWorldWallCollision(operator FlatlandAvatar, wall FlatlandWallFixture) FlatlandAvatar {
	switch wall.Orientation {
	case FlatlandWallX:
		return flatlandXWallCollision(operator, wall)
	case FlatlandWallY:
		return flatlandYWallCollision(operator, wall)
	default:
		return operator
	}
}

func flatlandXWallCollision(operator FlatlandAvatar, wall FlatlandWallFixture) FlatlandAvatar {
	x := operator.Location.X
	y := operator.Location.Y
	r := operator.Radius
	if wall.Y < y-r || wall.Y > y+r {
		return operator
	}
	if wall.XMin <= x && wall.XMax >= x {
		dy := r - (y - wall.Y)
		if y <= wall.Y {
			dy = -r - (y - wall.Y)
		}
		return flatlandTranslateAvatarWithoutEnergy(operator, 0, dy)
	}
	if x < wall.XMin {
		if math.Hypot(x-wall.XMin, y-wall.Y) < r {
			return FlatlandResist(FlatlandPoint{X: wall.XMin, Y: wall.Y}, operator)
		}
		return operator
	}
	if math.Hypot(x-wall.XMax, y-wall.Y) < r {
		return FlatlandResist(FlatlandPoint{X: wall.XMax, Y: wall.Y}, operator)
	}
	return operator
}

func flatlandYWallCollision(operator FlatlandAvatar, wall FlatlandWallFixture) FlatlandAvatar {
	x := operator.Location.X
	y := operator.Location.Y
	r := operator.Radius
	if wall.X < x-r || wall.X > x+r {
		return operator
	}
	if wall.YMin <= y && wall.YMax >= y {
		dx := r - (x - wall.X)
		if x <= wall.X {
			dx = -r - (x - wall.X)
		}
		return flatlandTranslateAvatarWithoutEnergy(operator, dx, 0)
	}
	if y < wall.YMin {
		if math.Hypot(y-wall.YMin, x-wall.X) < r {
			return FlatlandResist(FlatlandPoint{X: wall.X, Y: wall.YMin}, operator)
		}
		return operator
	}
	if math.Hypot(y-wall.YMax, x-wall.X) < r {
		return FlatlandResist(FlatlandPoint{X: wall.X, Y: wall.YMax}, operator)
	}
	return operator
}

func flatlandTranslateAvatarWithoutEnergy(avatar FlatlandAvatar, dx, dy float64) FlatlandAvatar {
	avatar.Location.X += dx
	avatar.Location.Y += dy
	avatar.Objects = flatlandTranslateObjects(avatar.Objects, dx, dy)
	return avatar
}
