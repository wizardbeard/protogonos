package scape

const FlatlandSectorSize = 10

// FlatlandLoc2Sector mirrors flatland:loc2sector/1 from the Erlang reference.
// The reference truncates X / 10 and Y / 10.
func FlatlandLoc2Sector(x, y float64) (int, int) {
	return int(x / FlatlandSectorSize), int(y / FlatlandSectorSize)
}
