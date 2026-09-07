package scape

import "math"

const (
	FlatlandScannerColorBlack  = -1
	FlatlandScannerColorCyan   = -0.75
	FlatlandScannerColorGreen  = -0.5
	FlatlandScannerColorYellow = -0.25
	FlatlandScannerColorBlue   = 0
	FlatlandScannerColorGrey   = 0.25
	FlatlandScannerColorRed    = 0.5
	FlatlandScannerColorBrown  = 0.75
	FlatlandScannerColorVoid   = 1
)

type flatlandRayHit struct {
	distance float64
	color    string
	energy   float64
}

func FlatlandDistanceScanner(density int, spread float64, loc, direction FlatlandPoint, avatars []FlatlandAvatar) []float64 {
	rays := FlatlandCreateUnitRays(direction, density, spread)
	out := make([]float64, 0, len(rays))
	for _, ray := range rays {
		hit := flatlandShortestIntersection(loc, ray, avatars)
		out = append(out, flatlandScannerDistance(hit.distance))
	}
	return out
}

func FlatlandColorScanner(density int, spread float64, loc, direction FlatlandPoint, avatars []FlatlandAvatar) []float64 {
	rays := FlatlandCreateUnitRays(direction, density, spread)
	out := make([]float64, 0, len(rays))
	for _, ray := range rays {
		hit := flatlandShortestIntersection(loc, ray, avatars)
		out = append(out, flatlandScannerColor(hit.distance, hit.color))
	}
	return out
}

func FlatlandEnergyScaner(density int, spread float64, loc, direction FlatlandPoint, avatars []FlatlandAvatar) []float64 {
	rays := FlatlandCreateUnitRays(direction, density, spread)
	out := make([]float64, 0, len(rays))
	for _, ray := range rays {
		hit := flatlandShortestIntersection(loc, ray, avatars)
		if hit.distance == math.Inf(1) || hit.distance == 0 {
			out = append(out, 0)
			continue
		}
		out = append(out, hit.energy/100)
	}
	return out
}

func FlatlandCreateUnitRays(direction FlatlandPoint, density int, spread float64) []FlatlandPoint {
	if density <= 0 {
		return nil
	}
	resolution := spread / float64(density)
	startAngle := -float64(density/2) * resolution
	if density%2 == 0 {
		startAngle += resolution / 2
	}
	rays := make([]FlatlandPoint, 0, density)
	for i := 0; i < density; i++ {
		angle := startAngle + float64(i)*resolution
		ray := FlatlandPoint{
			X: direction.X*math.Cos(angle) - direction.Y*math.Sin(angle),
			Y: direction.X*math.Sin(angle) + direction.Y*math.Cos(angle),
		}
		rays = append([]FlatlandPoint{ray}, rays...)
	}
	return rays
}

func flatlandShortestIntersection(loc, ray FlatlandPoint, avatars []FlatlandAvatar) flatlandRayHit {
	hit := flatlandRayHit{distance: math.Inf(1), color: "void"}
	for _, avatar := range avatars {
		next := flatlandAvatarIntersection(loc, ray, avatar, hit)
		if next.distance != hit.distance {
			next.energy = avatar.Energy
		} else {
			next.energy = hit.energy
		}
		hit = next
	}
	return hit
}

func flatlandAvatarIntersection(loc, ray FlatlandPoint, avatar FlatlandAvatar, hit flatlandRayHit) flatlandRayHit {
	for _, object := range avatar.Objects {
		switch object.Name {
		case "circle":
			hit = flatlandCircleIntersection(loc, ray, object, hit)
		case "line":
			hit = flatlandLineIntersection(loc, ray, object, hit)
		}
	}
	return hit
}

func flatlandCircleIntersection(loc, ray FlatlandPoint, object FlatlandObject, hit flatlandRayHit) flatlandRayHit {
	if len(object.Coords) == 0 {
		return hit
	}
	center := object.Coords[0]
	vx := loc.X - center.X
	vy := loc.Y - center.Y
	vdotd := vx*ray.X + vy*ray.Y
	discriminant := vdotd*vdotd - (vx*vx + vy*vy - object.Radius*object.Radius)
	if discriminant <= 0 {
		return hit
	}
	sqrtDiscriminant := math.Sqrt(discriminant)
	i1 := -vdotd - sqrtDiscriminant
	i2 := -vdotd + sqrtDiscriminant
	if i1 <= 0 || i2 <= 0 {
		return hit
	}
	result := math.Min(i1, i2)
	if result < hit.distance {
		hit.distance = result
		hit.color = object.Color
	}
	return hit
}

func flatlandLineIntersection(loc, ray FlatlandPoint, object FlatlandObject, hit flatlandRayHit) flatlandRayHit {
	if len(object.Coords) < 2 {
		return hit
	}
	a := object.Coords[0]
	b := object.Coords[1]
	perpXD1 := b.Y - a.Y
	perpYD1 := -(b.X - a.X)
	perpXD0 := ray.Y
	perpYD0 := -ray.X
	denom := perpXD1*ray.X + perpYD1*ray.Y
	if denom == 0 {
		return hit
	}
	rayLength := (perpXD1*(a.X-loc.X) + perpYD1*(a.Y-loc.Y)) / denom
	t := (perpXD0*(a.X-loc.X) + perpYD0*(a.Y-loc.Y)) / denom
	if rayLength >= 0 && t >= 0 && t <= 1 && rayLength < hit.distance {
		hit.distance = rayLength
		hit.color = object.Color
	}
	return hit
}

func flatlandScannerDistance(distance float64) float64 {
	if distance == math.Inf(1) || distance == 0 {
		return -1
	}
	return distance
}

func flatlandScannerColor(distance float64, color string) float64 {
	if distance == math.Inf(1) || distance == 0 {
		return FlatlandScannerColorVoid
	}
	return FlatlandScannerColorValue(color)
}

func FlatlandScannerColorValue(color string) float64 {
	switch color {
	case "black":
		return FlatlandScannerColorBlack
	case "cyan":
		return FlatlandScannerColorCyan
	case "green":
		return FlatlandScannerColorGreen
	case "yellow":
		return FlatlandScannerColorYellow
	case "blue":
		return FlatlandScannerColorBlue
	case "gret", "grey":
		return FlatlandScannerColorGrey
	case "red":
		return FlatlandScannerColorRed
	case "brown":
		return FlatlandScannerColorBrown
	default:
		return FlatlandScannerColorVoid
	}
}
