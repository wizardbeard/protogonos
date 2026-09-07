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

func TestFlatlandShortestDistanceMirrorsReferenceHelper(t *testing.T) {
	operator := FlatlandAvatar{ID: "operator", Location: FlatlandPoint{X: 2, Y: 3}}
	avatars := []FlatlandAvatar{
		{ID: "far", Location: FlatlandPoint{X: 12, Y: 3}},
		{ID: "near", Location: FlatlandPoint{X: 5, Y: 7}},
	}

	assertClose(t, "shortest distance", FlatlandShortestDistance(operator, avatars), 5)
	assertClose(t, "empty shortest distance", FlatlandShortestDistance(operator, nil), -1)
}

func TestFlatlandShortestIntrLineReturnsFormattedReferenceHit(t *testing.T) {
	plant := FlatlandCreatePlantAvatar("plant", FlatlandPoint{X: 10}, nil, FlatlandStateNoRespawn, FlatlandMetabolicStatic)
	poison := FlatlandCreatePoisonAvatar("poison", FlatlandPoint{X: 5}, nil, FlatlandStateNoRespawn, FlatlandMetabolicStatic)

	hit := FlatlandShortestIntrLine(FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandAvatar{plant, poison})
	assertClose(t, "distance", hit.Distance, 2)
	assertClose(t, "color value", hit.ColorValue, FlatlandScannerColorBlack)
	assertClose(t, "energy", hit.Energy, -2000)
	if hit.Color != FlatlandColorBlack {
		t.Fatalf("expected black hit color, got %+v", hit)
	}

	miss := FlatlandShortestIntrLine(FlatlandPoint{}, FlatlandPoint{X: -1}, []FlatlandAvatar{plant, poison})
	assertClose(t, "miss distance", miss.Distance, -1)
	assertClose(t, "miss color", miss.ColorValue, FlatlandScannerColorVoid)
	assertClose(t, "miss energy", miss.Energy, 0)
}

func TestFlatlandIntrReturnsRawObjectIntersection(t *testing.T) {
	object := flatlandCircle(FlatlandColorRed, FlatlandPoint{X: 10}, 3)

	hit := FlatlandIntr(FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandObject{object}, math.Inf(1), "void")
	assertClose(t, "raw distance", hit.Distance, 7)
	assertClose(t, "raw color value", hit.ColorValue, FlatlandScannerColorRed)
	if hit.Color != FlatlandColorRed {
		t.Fatalf("expected red raw hit color, got %+v", hit)
	}

	blocked := FlatlandIntr(FlatlandPoint{}, FlatlandPoint{X: 1}, []FlatlandObject{object}, 5, FlatlandColorBrown)
	assertClose(t, "blocked min distance", blocked.Distance, 5)
	assertClose(t, "blocked color value", blocked.ColorValue, FlatlandScannerColorBrown)
}
