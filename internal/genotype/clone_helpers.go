package genotype

import (
	"fmt"
	"strings"
)

// EvoHistoryEvent is a Go-native analog for variable-width evolutionary
// history tuples used in reference map_EvoHist flows.
type EvoHistoryEvent struct {
	Mutation string
	IDs      []string
}

// MapIDs mirrors genotype:map_ids/3 intent by generating a stable original->new
// ID mapping for a list of source IDs.
func MapIDs(ids []string, cloneID func(originalID string, index int) string) map[string]string {
	byOriginal := make(map[string]string, len(ids))
	used := make(map[string]struct{}, len(ids))
	for i, originalID := range ids {
		originalID = strings.TrimSpace(originalID)
		if originalID == "" {
			continue
		}
		if _, exists := byOriginal[originalID]; exists {
			continue
		}

		candidate := ""
		if cloneID != nil {
			candidate = strings.TrimSpace(cloneID(originalID, i))
		}
		if candidate == "" {
			candidate = defaultMappedID(originalID, i, used)
		}
		candidate = ensureUniqueID(candidate, originalID, i, used)
		byOriginal[originalID] = candidate
		used[candidate] = struct{}{}
	}
	return byOriginal
}

// MapEvoHistory mirrors genotype:map_EvoHist/2 by remapping event IDs with an
// ID map while preserving mutation ordering and tuple width semantics.
func MapEvoHistory(history []EvoHistoryEvent, idMap map[string]string) []EvoHistoryEvent {
	if len(history) == 0 {
		return nil
	}
	out := make([]EvoHistoryEvent, 0, len(history))
	for _, event := range history {
		cloned := EvoHistoryEvent{
			Mutation: event.Mutation,
		}
		if len(event.IDs) > 0 {
			cloned.IDs = make([]string, 0, len(event.IDs))
			for _, id := range event.IDs {
				if mapped, ok := idMap[id]; ok {
					cloned.IDs = append(cloned.IDs, mapped)
					continue
				}
				cloned.IDs = append(cloned.IDs, id)
			}
		}
		out = append(out, cloned)
	}
	return out
}

func defaultMappedID(originalID string, index int, used map[string]struct{}) string {
	base := sanitizeID(originalID)
	if base == "" {
		base = "id"
	}
	candidate := fmt.Sprintf("%s:clone:%d", base, index)
	if _, exists := used[candidate]; !exists {
		return candidate
	}
	for suffix := 1; ; suffix++ {
		candidate = fmt.Sprintf("%s:clone:%d:%d", base, index, suffix)
		if _, exists := used[candidate]; exists {
			continue
		}
		return candidate
	}
}

func ensureUniqueID(candidate, originalID string, index int, used map[string]struct{}) string {
	if candidate == "" {
		return defaultMappedID(originalID, index, used)
	}
	if _, exists := used[candidate]; !exists {
		return candidate
	}
	for suffix := 1; ; suffix++ {
		try := fmt.Sprintf("%s:%d", candidate, suffix)
		if _, exists := used[try]; exists {
			continue
		}
		return try
	}
}
