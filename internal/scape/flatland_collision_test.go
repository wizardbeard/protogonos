package scape

import "testing"

func TestFlatlandPushMatchesReferenceSeparationAndEnergyCost(t *testing.T) {
	pusher := testFlatlandCollisionAvatar("pusher", 0, 0, 3, 100)
	avatar := testFlatlandCollisionAvatar("avatar", 4, 0, 2, 50)

	got := FlatlandPush(pusher, avatar, 1)

	assertClose(t, "x", got.Location.X, 6)
	assertClose(t, "y", got.Location.Y, 0)
	assertClose(t, "energy", got.Energy, 40)
	assertClose(t, "object pivot x", got.Objects[0].Pivot.X, 6)
	assertClose(t, "object coord x", got.Objects[0].Coords[0].X, 7)
}

func TestFlatlandPushNoopsWhenPusherIsNotStronger(t *testing.T) {
	pusher := testFlatlandCollisionAvatar("pusher", 0, 0, 3, 10)
	avatar := testFlatlandCollisionAvatar("avatar", 4, 0, 2, 50)

	got := FlatlandPush(pusher, avatar, 1)

	assertClose(t, "x", got.Location.X, avatar.Location.X)
	assertClose(t, "energy", got.Energy, avatar.Energy)
}

func TestFlatlandResistMovesAvatarToRadiusFromOrigin(t *testing.T) {
	avatar := testFlatlandCollisionAvatar("avatar", 3, 4, 10, 50)

	got := FlatlandResist(FlatlandPoint{}, avatar)

	assertClose(t, "x", got.Location.X, 6)
	assertClose(t, "y", got.Location.Y, 8)
	assertClose(t, "energy", got.Energy, avatar.Energy)
	assertClose(t, "object pivot y", got.Objects[0].Pivot.Y, 8)
}

func TestFlatlandWorldWallCollisionXWallBody(t *testing.T) {
	avatar := testFlatlandCollisionAvatar("avatar", 150, 302, 10, 50)
	wall := flatlandXWall(300, 100, 200)

	got := FlatlandWorldWallCollision(avatar, wall)

	assertClose(t, "x", got.Location.X, 150)
	assertClose(t, "y", got.Location.Y, 310)
	assertClose(t, "energy", got.Energy, avatar.Energy)
}

func TestFlatlandWorldWallCollisionYWallBody(t *testing.T) {
	avatar := testFlatlandCollisionAvatar("avatar", 245, 300, 10, 50)
	wall := flatlandYWall(250, 100, 500)

	got := FlatlandWorldWallCollision(avatar, wall)

	assertClose(t, "x", got.Location.X, 240)
	assertClose(t, "y", got.Location.Y, 300)
	assertClose(t, "energy", got.Energy, avatar.Energy)
}

func TestFlatlandWorldWallCollisionEndpointUsesResist(t *testing.T) {
	avatar := testFlatlandCollisionAvatar("avatar", 95, 300, 10, 50)
	wall := flatlandXWall(300, 100, 200)

	got := FlatlandWorldWallCollision(avatar, wall)

	assertClose(t, "x", got.Location.X, 90)
	assertClose(t, "y", got.Location.Y, 300)
}

func TestFlatlandWorldWallCollisionNoopsOutsideRadius(t *testing.T) {
	avatar := testFlatlandCollisionAvatar("avatar", 150, 320, 10, 50)
	wall := flatlandXWall(300, 100, 200)

	got := FlatlandWorldWallCollision(avatar, wall)

	assertClose(t, "x", got.Location.X, avatar.Location.X)
	assertClose(t, "y", got.Location.Y, avatar.Location.Y)
}

func testFlatlandCollisionAvatar(id string, x, y, radius, energy float64) FlatlandAvatar {
	return FlatlandAvatar{
		Type:     "prey",
		Location: FlatlandPoint{X: x, Y: y},
		Radius:   radius,
		Energy:   energy,
		Objects: []FlatlandObject{
			{
				Name:  "circle",
				ID:    id,
				Pivot: FlatlandPoint{X: x, Y: y},
				Coords: []FlatlandPoint{
					{X: x + 1, Y: y},
				},
			},
		},
	}
}
