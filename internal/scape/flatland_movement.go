package scape

import "math"

const FlatlandRotationRatio = math.Pi / 4

type FlatlandPoint struct {
	X float64
	Y float64
}

type FlatlandObject struct {
	Name      string
	ID        string
	Color     string
	Pivot     FlatlandPoint
	Coords    []FlatlandPoint
	Parameter string
}

type FlatlandAvatar struct {
	Type      string
	Location  FlatlandPoint
	Direction FlatlandPoint
	Energy    float64
	Age       int
	Objects   []FlatlandObject
}

func FlatlandMove(avatar FlatlandAvatar, speed float64) FlatlandAvatar {
	dx, dy := flatlandMoveDelta(avatar, speed)
	avatar.Energy -= 0.1*math.Sqrt(dx*dx+dy*dy) + 0.1
	avatar.Location.X += dx
	avatar.Location.Y += dy
	avatar.Objects = flatlandTranslateObjects(avatar.Objects, dx, dy)
	return avatar
}

func FlatlandTranslate(avatar FlatlandAvatar, delta FlatlandPoint) FlatlandAvatar {
	avatar.Energy -= 0.1*math.Hypot(delta.X, delta.Y) + 0.1
	avatar.Location.X += delta.X
	avatar.Location.Y += delta.Y
	avatar.Objects = flatlandTranslateObjects(avatar.Objects, delta.X, delta.Y)
	return avatar
}

func FlatlandRotate(avatar FlatlandAvatar, controlAngle float64) FlatlandAvatar {
	angle := controlAngle * FlatlandRotationRatio
	avatar.Energy -= 0.1*math.Abs(angle) + 0.1
	dx := avatar.Direction.X
	dy := avatar.Direction.Y
	avatar.Direction = FlatlandPoint{
		X: dx*math.Cos(angle) - dy*math.Sin(angle),
		Y: dx*math.Sin(angle) + dy*math.Cos(angle),
	}
	avatar.Objects = flatlandRotateObjects(avatar.Objects, angle)
	return avatar
}

func FlatlandTwoWheels(avatar FlatlandAvatar, rightWheel, leftWheel float64) FlatlandAvatar {
	speed, angle := FlatlandTwoWheelToMoveRotate(rightWheel, leftWheel)
	avatar = FlatlandRotate(avatar, angle)
	avatar = FlatlandMove(avatar, speed)
	avatar.Age++
	return avatar
}

func FlatlandMoveAndRotate(avatar FlatlandAvatar, speed, angle float64) FlatlandAvatar {
	return FlatlandRotate(FlatlandMove(avatar, speed), angle)
}

func FlatlandRotateAndMove(avatar FlatlandAvatar, speed, angle float64) FlatlandAvatar {
	return FlatlandMove(FlatlandRotate(avatar, angle), speed)
}

func FlatlandRotateAndTranslate(avatar FlatlandAvatar, translation FlatlandPoint, angle float64) FlatlandAvatar {
	return FlatlandTranslate(FlatlandRotate(avatar, angle), translation)
}

func FlatlandTwoWheelToMoveRotate(rightWheel, leftWheel float64) (speed, angle float64) {
	return (rightWheel + leftWheel) / 2, rightWheel - leftWheel
}

func flatlandMoveDelta(avatar FlatlandAvatar, speed float64) (float64, float64) {
	if avatar.Type != "prey" {
		speed *= 0.9
	}
	return avatar.Direction.X * speed, avatar.Direction.Y * speed
}

func flatlandTranslateObjects(objects []FlatlandObject, dx, dy float64) []FlatlandObject {
	out := make([]FlatlandObject, len(objects))
	for i, object := range objects {
		out[i] = object
		out[i].Pivot.X += dx
		out[i].Pivot.Y += dy
		out[i].Coords = make([]FlatlandPoint, len(object.Coords))
		for j, coord := range object.Coords {
			out[i].Coords[j] = FlatlandPoint{X: coord.X + dx, Y: coord.Y + dy}
		}
	}
	return out
}

func flatlandRotateObjects(objects []FlatlandObject, angle float64) []FlatlandObject {
	out := make([]FlatlandObject, len(objects))
	for i, object := range objects {
		out[i] = object
		out[i].Coords = make([]FlatlandPoint, len(object.Coords))
		for j, coord := range object.Coords {
			x := coord.X - object.Pivot.X
			y := coord.Y - object.Pivot.Y
			out[i].Coords[j] = FlatlandPoint{
				X: x*math.Cos(angle) - y*math.Sin(angle) + object.Pivot.X,
				Y: x*math.Sin(angle) + y*math.Cos(angle) + object.Pivot.Y,
			}
		}
	}
	return out
}
