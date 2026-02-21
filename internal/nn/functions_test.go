package nn

import (
	"math"
	"testing"
)

func TestSaturationHelpers(t *testing.T) {
	if got := Saturation(1500); got != 1000 {
		t.Fatalf("expected saturation upper clamp, got=%f", got)
	}
	if got := Saturation(-1500); got != -1000 {
		t.Fatalf("expected saturation lower clamp, got=%f", got)
	}
	if got := SaturationWithSpread(5, 2); got != 2 {
		t.Fatalf("expected spread clamp, got=%f", got)
	}
	if got := SaturationWithSpread(-5, 2); got != -2 {
		t.Fatalf("expected spread lower clamp, got=%f", got)
	}
}

func TestScaleAndSatHelpers(t *testing.T) {
	if got := ScaleValue(2, 4, 0); math.Abs(got-0) > 1e-12 {
		t.Fatalf("expected midpoint scale=0, got=%f", got)
	}
	gotSlice := ScaleSlice([]float64{0, 2, 4}, 4, 0)
	wantSlice := []float64{-1, 0, 1}
	for i := range wantSlice {
		if math.Abs(gotSlice[i]-wantSlice[i]) > 1e-12 {
			t.Fatalf("unexpected scaled slice at %d: got=%f want=%f", i, gotSlice[i], wantSlice[i])
		}
	}
	if got := Sat(5, 3, -3); got != 3 {
		t.Fatalf("expected sat max clamp, got=%f", got)
	}
	if got := SatDeadZone(0.1, 3, -3, 0.5, -0.5); got != 0 {
		t.Fatalf("expected sat deadzone zero, got=%f", got)
	}
}

func TestAvgAndStd(t *testing.T) {
	avg, err := Avg([]float64{1, 2, 3})
	if err != nil {
		t.Fatalf("avg failed: %v", err)
	}
	if math.Abs(avg-2) > 1e-12 {
		t.Fatalf("unexpected avg: %f", avg)
	}
	std, err := Std([]float64{1, 2, 3})
	if err != nil {
		t.Fatalf("std failed: %v", err)
	}
	if math.Abs(std-math.Sqrt(2.0/3.0)) > 1e-12 {
		t.Fatalf("unexpected std: %f", std)
	}
	if _, err := Avg(nil); err == nil {
		t.Fatal("expected avg empty error")
	}
}

func TestVectorHelpers(t *testing.T) {
	diff, err := VectorDifference([]float64{1, 2}, []float64{3, 5})
	if err != nil {
		t.Fatalf("vector difference failed: %v", err)
	}
	if diff[0] != 2 || diff[1] != 3 {
		t.Fatalf("unexpected vector difference: %v", diff)
	}
	d, err := Distance([]float64{1, 2}, []float64{4, 6})
	if err != nil {
		t.Fatalf("distance failed: %v", err)
	}
	if math.Abs(d-5) > 1e-12 {
		t.Fatalf("unexpected distance: %f", d)
	}
	if _, err := Distance([]float64{1}, []float64{1, 2}); err == nil {
		t.Fatal("expected distance mismatch error")
	}
}

func TestCoordinateTransformsAndFeatures(t *testing.T) {
	c := Cartesian([]float64{1, 2}, []float64{3, 4})
	if len(c) != 4 || c[0] != 1 || c[3] != 4 {
		t.Fatalf("unexpected cartesian feature concat: %v", c)
	}

	polar, err := Polar([]float64{0, 1}, []float64{1, 0})
	if err != nil {
		t.Fatalf("polar features failed: %v", err)
	}
	if len(polar) != 4 {
		t.Fatalf("unexpected polar feature length: %d", len(polar))
	}

	spherical, err := Spherical([]float64{0, 0, 1}, []float64{0, 1, 0})
	if err != nil {
		t.Fatalf("spherical features failed: %v", err)
	}
	if len(spherical) != 6 {
		t.Fatalf("unexpected spherical feature length: %d", len(spherical))
	}

	cd, err := CartesianDistance([]float64{0, 0}, []float64{3, 4})
	if err != nil {
		t.Fatalf("cartesian distance failed: %v", err)
	}
	if len(cd) != 1 || math.Abs(cd[0]-5) > 1e-12 {
		t.Fatalf("unexpected cartesian distance: %v", cd)
	}

	cgd, err := CartesianGaussedCoordDiffs([]float64{0, 0}, []float64{1, -1})
	if err != nil {
		t.Fatalf("gaussed diffs failed: %v", err)
	}
	if len(cgd) != 2 {
		t.Fatalf("unexpected gaussed diffs length: %d", len(cgd))
	}
}

func TestCoordinateConversions(t *testing.T) {
	cart, err := ToCartesian("spherical", []float64{1, 0, math.Pi / 2})
	if err != nil {
		t.Fatalf("to cartesian spherical failed: %v", err)
	}
	if len(cart) != 3 || math.Abs(cart[0]-1) > 1e-12 || math.Abs(cart[1]) > 1e-12 || math.Abs(cart[2]) > 1e-12 {
		t.Fatalf("unexpected spherical->cartesian: %v", cart)
	}

	cart, err = ToCartesian("polar", []float64{2, math.Pi / 2})
	if err != nil {
		t.Fatalf("to cartesian polar failed: %v", err)
	}
	if len(cart) != 3 || math.Abs(cart[0]) > 1e-12 || math.Abs(cart[1]-2) > 1e-12 {
		t.Fatalf("unexpected polar->cartesian: %v", cart)
	}

	pol, err := CartesianToPolarList([]float64{0, 1, 0})
	if err != nil {
		t.Fatalf("cartesian->polar failed: %v", err)
	}
	if len(pol) != 2 || math.Abs(pol[0]-1) > 1e-12 {
		t.Fatalf("unexpected cartesian->polar: %v", pol)
	}

	sph, err := CartesianToSphericalList([]float64{0, 0, 1})
	if err != nil {
		t.Fatalf("cartesian->spherical failed: %v", err)
	}
	if len(sph) != 3 || math.Abs(sph[0]-1) > 1e-12 {
		t.Fatalf("unexpected cartesian->spherical: %v", sph)
	}
}

func TestNormalizeAndIOW(t *testing.T) {
	n, err := Normalize([]float64{3, 4})
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}
	if math.Abs(n[0]-0.6) > 1e-12 || math.Abs(n[1]-0.8) > 1e-12 {
		t.Fatalf("unexpected normalized vector: %v", n)
	}
	if _, err := Normalize([]float64{0, 0}); err == nil {
		t.Fatal("expected normalize zero vector error")
	}

	out := IOW(nil, nil, []float64{1, 2, 3})
	if len(out) != 3 || out[0] != 1 || out[2] != 3 {
		t.Fatalf("unexpected iow passthrough: %v", out)
	}

	withIOW := CartesianWithIOW([]float64{1}, []float64{2}, []float64{9, 8, 7})
	if len(withIOW) != 5 || withIOW[0] != 9 || withIOW[3] != 1 || withIOW[4] != 2 {
		t.Fatalf("unexpected cartesian with iow: %v", withIOW)
	}
}

func TestXORDemoStep(t *testing.T) {
	out1, out2, next := XORDemoStep(1, -1, 0)
	if math.Abs(next-out1) > 1e-12 {
		t.Fatalf("expected recurrent output to track output1, out1=%f next=%f", out1, next)
	}
	if math.IsNaN(out2) || math.IsInf(out2, 0) {
		t.Fatalf("unexpected invalid xor demo output2: %f", out2)
	}
}
