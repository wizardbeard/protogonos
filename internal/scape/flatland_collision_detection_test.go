package scape

import "testing"

func TestFlatlandCollisionDetectionDestroyOmitsTargetAndCountsKill(t *testing.T) {
	operator := FlatlandAvatar{
		Type:      FlatlandObjectPredator,
		ID:        "pred",
		Location:  FlatlandPoint{},
		Direction: FlatlandPoint{X: 1},
		Radius:    3,
		Energy:    1000,
	}
	prey := FlatlandAvatar{
		Type:     FlatlandObjectPrey,
		ID:       "prey",
		Location: FlatlandPoint{X: 4},
		Radius:   1,
		Energy:   100,
	}

	result := FlatlandCollisionDetection(operator, []FlatlandAvatar{operator, prey})
	got, ok := findFlatlandAvatar(result.Avatars, "pred")
	if !ok {
		t.Fatalf("operator missing from avatar list: %+v", result.Avatars)
	}
	if _, ok := findFlatlandAvatar(result.Avatars, "prey"); ok {
		t.Fatalf("destroyed prey should be omitted: %+v", result.Avatars)
	}
	if result.Kills != 1 || got.Kills != 1 {
		t.Fatalf("expected one kill, result=%+v operator=%+v", result, got)
	}
	assertClose(t, "operator energy", got.Energy, 1500)
}

func TestFlatlandCollisionDetectionPlantRespawnAccumulatesEnergyAndKill(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, ID: "prey", Location: FlatlandPoint{}, Radius: 3, Energy: 9000}
	plant := FlatlandAvatar{
		Type:     FlatlandObjectPlant,
		ID:       "plant",
		Location: FlatlandPoint{X: 1},
		Radius:   3,
		Energy:   500,
		State:    FlatlandStateRespawn,
	}

	result := FlatlandCollisionDetection(operator, []FlatlandAvatar{operator, plant})
	got, ok := findFlatlandAvatar(result.Avatars, "prey")
	if !ok {
		t.Fatalf("operator missing from avatar list: %+v", result.Avatars)
	}
	respawned, ok := findFlatlandAvatar(result.Avatars, "plant")
	if !ok {
		t.Fatalf("respawned plant missing: %+v", result.Avatars)
	}
	if result.Kills != 1 || got.Kills != 1 {
		t.Fatalf("expected plant kill score, result=%+v operator=%+v", result, got)
	}
	assertClose(t, "operator energy", got.Energy, 9500)
	assertClose(t, "respawned plant energy", respawned.Energy, 500)
	if respawned.State != FlatlandStateRespawn || respawned.Objects[0].Color != FlatlandColorGreen {
		t.Fatalf("unexpected respawned plant: %+v", respawned)
	}
}

func TestFlatlandCollisionDetectionPoisonNoRespawnNoKill(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, ID: "prey", Location: FlatlandPoint{}, Radius: 3, Energy: 1000}
	poison := FlatlandAvatar{
		Type:     FlatlandObjectPoison,
		ID:       "poison",
		Location: FlatlandPoint{X: 1},
		Radius:   3,
		Energy:   -2000,
		State:    FlatlandStateNoRespawn,
	}

	result := FlatlandCollisionDetection(operator, []FlatlandAvatar{operator, poison})
	got, ok := findFlatlandAvatar(result.Avatars, "prey")
	if !ok {
		t.Fatalf("operator missing from avatar list: %+v", result.Avatars)
	}
	if _, ok := findFlatlandAvatar(result.Avatars, "poison"); ok {
		t.Fatalf("no-respawn poison should be omitted: %+v", result.Avatars)
	}
	if result.Kills != 0 || got.Kills != 0 {
		t.Fatalf("poison should not count as a kill, result=%+v operator=%+v", result, got)
	}
	assertClose(t, "operator energy", got.Energy, -1000)
}

func TestFlatlandCollisionDetectionWallUpdatesOperatorAndRetainsWall(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, ID: "prey", Location: FlatlandPoint{X: 150, Y: 298}, Radius: 5, Energy: 1000}
	wall := FlatlandCreateWallAvatar("wall", FlatlandWallFixture{
		Type:        FlatlandObjectWall,
		Orientation: FlatlandWallX,
		X:           150,
		Y:           300,
		XMin:        100,
		XMax:        200,
		YMin:        300,
		YMax:        300,
		Energy:      10000,
	})

	result := FlatlandCollisionDetection(operator, []FlatlandAvatar{operator, wall})
	got, ok := findFlatlandAvatar(result.Avatars, "prey")
	if !ok {
		t.Fatalf("operator missing from avatar list: %+v", result.Avatars)
	}
	if _, ok := findFlatlandAvatar(result.Avatars, "wall"); !ok {
		t.Fatalf("wall should be retained: %+v", result.Avatars)
	}
	assertClose(t, "operator y", got.Location.Y, 295)
	assertClose(t, "operator energy", got.Energy, 1000)
}

func TestFlatlandCollisionDetectionSaturatesOperatorEnergy(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, ID: "prey", Location: FlatlandPoint{}, Radius: 3, Energy: 9900}
	plant := FlatlandAvatar{
		Type:     FlatlandObjectPlant,
		ID:       "plant",
		Location: FlatlandPoint{X: 1},
		Radius:   3,
		Energy:   500,
		State:    FlatlandStateNoRespawn,
	}

	result := FlatlandCollisionDetection(operator, []FlatlandAvatar{operator, plant})
	got, ok := findFlatlandAvatar(result.Avatars, "prey")
	if !ok {
		t.Fatalf("operator missing from avatar list: %+v", result.Avatars)
	}
	assertClose(t, "operator energy", got.Energy, FlatlandMaxEnergy)
}

func findFlatlandAvatar(avatars []FlatlandAvatar, id string) (FlatlandAvatar, bool) {
	for _, avatar := range avatars {
		if avatar.ID == id {
			return avatar, true
		}
	}
	return FlatlandAvatar{}, false
}
