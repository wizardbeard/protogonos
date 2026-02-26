package scapeid

import "strings"

// Normalize canonicalizes scape names and reference aliases.
func Normalize(name string) string {
	normalized := strings.TrimSpace(strings.ToLower(name))
	normalized = strings.ReplaceAll(normalized, "_", "-")
	normalized = strings.ReplaceAll(normalized, " ", "-")
	normalized = strings.Trim(normalized, "-")
	if normalized == "" {
		return ""
	}
	if canonical, ok := normalizeKnownAlias(normalized); ok {
		return canonical
	}
	return normalized
}

func normalizeKnownAlias(normalized string) (string, bool) {
	for _, candidate := range aliasCandidates(normalized) {
		if canonical, ok := canonicalScapeName(candidate); ok {
			return canonical, true
		}
	}
	return "", false
}

func aliasCandidates(normalized string) []string {
	candidate := strings.TrimPrefix(normalized, "scape-")
	if candidate == normalized {
		candidate = strings.TrimPrefix(candidate, "scape")
	}
	candidate = strings.Trim(candidate, "-")

	candidates := []string{normalized}
	if candidate != "" && candidate != normalized {
		candidates = append(candidates, candidate)
	}

	trimmedCandidate := trimSimSuffix(candidate)
	if trimmedCandidate != "" && trimmedCandidate != candidate {
		candidates = append(candidates, trimmedCandidate)
	}

	trimmedNormalized := trimSimSuffix(normalized)
	if trimmedNormalized != "" &&
		trimmedNormalized != normalized &&
		trimmedNormalized != candidate &&
		trimmedNormalized != trimmedCandidate {
		candidates = append(candidates, trimmedNormalized)
	}
	return candidates
}

func trimSimSuffix(value string) string {
	switch {
	case strings.HasSuffix(value, "-sim1"):
		return strings.TrimSuffix(value, "-sim1")
	case strings.HasSuffix(value, "sim1") && !strings.Contains(value, "-"):
		return strings.TrimSuffix(value, "sim1")
	case strings.HasSuffix(value, "-sim"):
		return strings.TrimSuffix(value, "-sim")
	case strings.HasSuffix(value, "sim") && !strings.Contains(value, "-"):
		return strings.TrimSuffix(value, "sim")
	default:
		return value
	}
}

func canonicalScapeName(alias string) (string, bool) {
	switch alias {
	case "xor":
		return "xor", true
	case "regression-mimic":
		return "regression-mimic", true
	case "cart-pole-lite":
		return "cart-pole-lite", true
	case "pole2-balancing":
		return "pole2-balancing", true
	case "dtm":
		return "dtm", true
	case "flatland":
		return "flatland", true
	case "gtsa":
		return "gtsa", true
	case "fx":
		return "fx", true
	case "epitopes":
		return "epitopes", true
	case "llvm-phase-ordering":
		return "llvm-phase-ordering", true
	}

	compact := strings.ReplaceAll(alias, "-", "")
	switch compact {
	case "xor":
		return "xor", true
	case "regressionmimic":
		return "regression-mimic", true
	case "cartpolelite":
		return "cart-pole-lite", true
	case "pole2balancing", "pb":
		return "pole2-balancing", true
	case "dtm":
		return "dtm", true
	case "flatland":
		return "flatland", true
	case "gtsa":
		return "gtsa", true
	case "fx":
		return "fx", true
	case "epitopes":
		return "epitopes", true
	case "llvmphaseordering":
		return "llvm-phase-ordering", true
	default:
		return "", false
	}
}
