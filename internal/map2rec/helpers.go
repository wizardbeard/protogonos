package map2rec

import "errors"

var ErrUnsupportedKind = errors.New("unsupported map2rec kind")

func asString(v any) (string, bool) {
	switch x := v.(type) {
	case string:
		return x, true
	default:
		return "", false
	}
}

func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case int8:
		return int(x), true
	case int16:
		return int(x), true
	case int32:
		return int(x), true
	case int64:
		return int(x), true
	case float64:
		return int(x), true
	case float32:
		return int(x), true
	default:
		return 0, false
	}
}

func asFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int8:
		return float64(x), true
	case int16:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	default:
		return 0, false
	}
}

func asBool(v any) (bool, bool) {
	switch x := v.(type) {
	case bool:
		return x, true
	default:
		return false, false
	}
}

func asStrings(v any) ([]string, bool) {
	switch xs := v.(type) {
	case []string:
		return append([]string(nil), xs...), true
	case []any:
		out := make([]string, 0, len(xs))
		for _, item := range xs {
			s, ok := asString(item)
			if !ok {
				return nil, false
			}
			out = append(out, s)
		}
		return out, true
	default:
		return nil, false
	}
}

func asFloat64s(v any) ([]float64, bool) {
	switch xs := v.(type) {
	case []float64:
		return append([]float64(nil), xs...), true
	case []any:
		out := make([]float64, 0, len(xs))
		for _, item := range xs {
			f, ok := asFloat64(item)
			if !ok {
				return nil, false
			}
			out = append(out, f)
		}
		return out, true
	default:
		return nil, false
	}
}

func asAnySlice(v any) ([]any, bool) {
	switch xs := v.(type) {
	case []any:
		return append([]any(nil), xs...), true
	case []string:
		out := make([]any, 0, len(xs))
		for _, item := range xs {
			out = append(out, item)
		}
		return out, true
	case []int:
		out := make([]any, 0, len(xs))
		for _, item := range xs {
			out = append(out, item)
		}
		return out, true
	case []float64:
		out := make([]any, 0, len(xs))
		for _, item := range xs {
			out = append(out, item)
		}
		return out, true
	default:
		return nil, false
	}
}

func asDurationSpec(v any) (DurationSpec, bool) {
	switch x := v.(type) {
	case map[string]any:
		name, ok1 := asString(x["name"])
		param, ok2 := asFloat64(x["param"])
		if !ok1 || !ok2 {
			return DurationSpec{}, false
		}
		return DurationSpec{Name: name, Param: param}, true
	case []any:
		if len(x) != 2 {
			return DurationSpec{}, false
		}
		name, ok1 := asString(x[0])
		param, ok2 := asFloat64(x[1])
		if !ok1 || !ok2 {
			return DurationSpec{}, false
		}
		return DurationSpec{Name: name, Param: param}, true
	default:
		return DurationSpec{}, false
	}
}

func asWeightedOperators(v any) ([]WeightedOperator, bool) {
	raw, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]WeightedOperator, 0, len(raw))
	for _, item := range raw {
		switch x := item.(type) {
		case map[string]any:
			name, ok1 := asString(x["name"])
			weight, ok2 := asFloat64(x["weight"])
			if !ok1 || !ok2 {
				return nil, false
			}
			out = append(out, WeightedOperator{Name: name, Weight: weight})
		case []any:
			if len(x) != 2 {
				return nil, false
			}
			name, ok1 := asString(x[0])
			weight, ok2 := asFloat64(x[1])
			if !ok1 || !ok2 {
				return nil, false
			}
			out = append(out, WeightedOperator{Name: name, Weight: weight})
		default:
			return nil, false
		}
	}
	return out, true
}

func asMutationCountPolicies(v any) ([]MutationCountPolicy, bool) {
	raw, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]MutationCountPolicy, 0, len(raw))
	for _, item := range raw {
		switch x := item.(type) {
		case map[string]any:
			name, ok1 := asString(x["name"])
			param, ok2 := asFloat64(x["param"])
			if !ok1 || !ok2 {
				return nil, false
			}
			out = append(out, MutationCountPolicy{Name: name, Param: param})
		case []any:
			if len(x) != 2 {
				return nil, false
			}
			name, ok1 := asString(x[0])
			param, ok2 := asFloat64(x[1])
			if !ok1 || !ok2 {
				return nil, false
			}
			out = append(out, MutationCountPolicy{Name: name, Param: param})
		default:
			return nil, false
		}
	}
	return out, true
}
