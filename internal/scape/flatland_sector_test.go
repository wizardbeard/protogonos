package scape

import "testing"

func TestFlatlandLoc2SectorMatchesReferenceTruncation(t *testing.T) {
	tests := []struct {
		name  string
		x, y  float64
		wantX int
		wantY int
	}{
		{name: "origin", x: 0, y: 0, wantX: 0, wantY: 0},
		{name: "positive", x: 99.9, y: 101.2, wantX: 9, wantY: 10},
		{name: "negative truncates toward zero", x: -19.9, y: -20.1, wantX: -1, wantY: -2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotX, gotY := FlatlandLoc2Sector(tc.x, tc.y)
			if gotX != tc.wantX || gotY != tc.wantY {
				t.Fatalf("expected sector=(%d,%d), got=(%d,%d)", tc.wantX, tc.wantY, gotX, gotY)
			}
		})
	}
}
