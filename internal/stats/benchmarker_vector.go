package stats

func BenchmarkerVectorGT(v1, v2 []float64) bool {
	if v2 == nil || len(v1) != len(v2) {
		return false
	}
	acc := 0.0
	for i := range v1 {
		if v1[i] < v2[i] {
			return false
		}
		acc += v1[i] - v2[i]
	}
	return acc > 0
}

func BenchmarkerVectorLT(v1, v2 []float64) bool {
	if v2 == nil || len(v1) != len(v2) {
		return false
	}
	acc := 0.0
	for i := range v1 {
		if v1[i] > v2[i] {
			return false
		}
		acc += v1[i] - v2[i]
	}
	return acc < 0
}

func BenchmarkerVectorEQ(v1, v2 []float64) bool {
	if v2 == nil || len(v1) != len(v2) {
		return false
	}
	for i := range v1 {
		if v1[i] != v2[i] {
			return false
		}
	}
	return true
}
