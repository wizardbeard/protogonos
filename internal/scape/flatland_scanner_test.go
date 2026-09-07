package scape

import (
	"math"
	"testing"
)

func TestFlatlandCreateUnitRaysMirrorsReferenceOrder(t *testing.T) {
	rays := FlatlandCreateUnitRays(FlatlandPoint{X: 1}, 3, math.Pi/2)
	if len(rays) != 3 {
		t.Fatalf("expected 3 rays, got %d", len(rays))
	}
	assertClose(t, "first ray x", rays[0].X, math.Cos(math.Pi/6))
	assertClose(t, "first ray y", rays[0].Y, math.Sin(math.Pi/6))
	assertClose(t, "middle ray x", rays[1].X, 1)
	assertClose(t, "middle ray y", rays[1].Y, 0)
	assertClose(t, "last ray x", rays[2].X, math.Cos(-math.Pi/6))
	assertClose(t, "last ray y", rays[2].Y, math.Sin(-math.Pi/6))
}

func TestFlatlandScannersHitNearestCircle(t *testing.T) {
	plant := FlatlandCreatePlantAvatar("plant", FlatlandPoint{X: 10}, nil, FlatlandStateNoRespawn, FlatlandMetabolicStatic)

	distance := FlatlandDistanceScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})
	color := FlatlandColorScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})
	energy := FlatlandEnergyScaner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})

	assertClose(t, "distance", distance[0], 7)
	assertClose(t, "color", color[0], FlatlandScannerColorGreen)
	assertClose(t, "energy", energy[0], 5)
}

func TestFlatlandScannersReturnReferenceMissValues(t *testing.T) {
	plant := FlatlandCreatePlantAvatar("plant", FlatlandPoint{X: -10}, nil, FlatlandStateNoRespawn, FlatlandMetabolicStatic)

	distance := FlatlandDistanceScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})
	color := FlatlandColorScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})
	energy := FlatlandEnergyScaner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant})

	assertClose(t, "distance", distance[0], -1)
	assertClose(t, "color", color[0], FlatlandScannerColorVoid)
	assertClose(t, "energy", energy[0], 0)
}

func TestFlatlandScannersHitLineObject(t *testing.T) {
	wall := FlatlandCreateWallAvatar("wall", FlatlandWallFixture{
		Type:        FlatlandObjectWall,
		Orientation: FlatlandWallY,
		X:           5,
		Y:           0,
		XMin:        5,
		XMax:        5,
		YMin:        -5,
		YMax:        5,
		Energy:      10000,
	})

	distance := FlatlandDistanceScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{wall})
	color := FlatlandColorScanner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{wall})
	energy := FlatlandEnergyScaner(1, 1, FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{wall})

	assertClose(t, "distance", distance[0], 5)
	assertClose(t, "color", color[0], FlatlandScannerColorBrown)
	assertClose(t, "energy", energy[0], 100)
}

func TestFlatlandScannerColorValueMirrorsReferenceMapping(t *testing.T) {
	cases := map[string]float64{
		"black":  FlatlandScannerColorBlack,
		"cyan":   FlatlandScannerColorCyan,
		"green":  FlatlandScannerColorGreen,
		"yellow": FlatlandScannerColorYellow,
		"blue":   FlatlandScannerColorBlue,
		"gret":   FlatlandScannerColorGrey,
		"grey":   FlatlandScannerColorGrey,
		"red":    FlatlandScannerColorRed,
		"brown":  FlatlandScannerColorBrown,
		"other":  FlatlandScannerColorVoid,
	}
	for color, want := range cases {
		assertClose(t, color, FlatlandScannerColorValue(color), want)
	}
}
