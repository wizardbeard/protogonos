package scape

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// FlatlandOverrides configures optional per-run flatland parameter overrides.
// Zero-values keep the selected mode defaults.
type FlatlandOverrides struct {
	ScannerProfile     string
	ScannerSpread      *float64
	ScannerOffset      *float64
	RandomizeLayout    *bool
	LayoutVariants     *int
	ForceLayoutVariant *int
}

type flatlandOverrides struct {
	scannerProfile     string
	hasScannerProfile  bool
	scannerSpread      float64
	hasScannerSpread   bool
	scannerOffset      float64
	hasScannerOffset   bool
	randomizeLayout    bool
	hasRandomizeLayout bool
	layoutVariants     int
	hasLayoutVariants  bool
	forcedLayout       int
	hasForcedLayout    bool
}

type flatlandOverridesContextKey struct{}

// WithFlatlandOverrides returns a context carrying optional per-run flatland overrides.
func WithFlatlandOverrides(ctx context.Context, overrides FlatlandOverrides) (context.Context, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	normalized, err := normalizeFlatlandOverrides(overrides)
	if err != nil {
		return nil, err
	}
	return context.WithValue(ctx, flatlandOverridesContextKey{}, normalized), nil
}

func flatlandOverridesFromContext(ctx context.Context) (flatlandOverrides, bool) {
	if ctx == nil {
		return flatlandOverrides{}, false
	}
	overrides, ok := ctx.Value(flatlandOverridesContextKey{}).(flatlandOverrides)
	if !ok {
		return flatlandOverrides{}, false
	}
	return overrides, true
}

func normalizeFlatlandOverrides(raw FlatlandOverrides) (flatlandOverrides, error) {
	normalized := flatlandOverrides{}

	if strings.TrimSpace(raw.ScannerProfile) != "" {
		profile, err := resolveFlatlandScannerProfile(raw.ScannerProfile)
		if err != nil {
			return flatlandOverrides{}, err
		}
		normalized.scannerProfile = profile
		normalized.hasScannerProfile = true
	}

	if raw.ScannerSpread != nil {
		spread := *raw.ScannerSpread
		if math.IsNaN(spread) || math.IsInf(spread, 0) {
			return flatlandOverrides{}, fmt.Errorf("flatland scanner spread must be finite")
		}
		if spread < 0.05 || spread > 1 {
			return flatlandOverrides{}, fmt.Errorf("flatland scanner spread must be within [0.05, 1], got %f", spread)
		}
		normalized.scannerSpread = spread
		normalized.hasScannerSpread = true
	}

	if raw.ScannerOffset != nil {
		offset := *raw.ScannerOffset
		if math.IsNaN(offset) || math.IsInf(offset, 0) {
			return flatlandOverrides{}, fmt.Errorf("flatland scanner offset must be finite")
		}
		if offset < -1 || offset > 1 {
			return flatlandOverrides{}, fmt.Errorf("flatland scanner offset must be within [-1, 1], got %f", offset)
		}
		normalized.scannerOffset = offset
		normalized.hasScannerOffset = true
	}

	if raw.RandomizeLayout != nil {
		normalized.randomizeLayout = *raw.RandomizeLayout
		normalized.hasRandomizeLayout = true
	}

	if raw.LayoutVariants != nil {
		variants := *raw.LayoutVariants
		if variants <= 0 {
			return flatlandOverrides{}, fmt.Errorf("flatland layout variants must be > 0, got %d", variants)
		}
		normalized.layoutVariants = variants
		normalized.hasLayoutVariants = true
	}

	if raw.ForceLayoutVariant != nil {
		normalized.forcedLayout = *raw.ForceLayoutVariant
		normalized.hasForcedLayout = true
	}

	return normalized, nil
}

func applyFlatlandOverrides(cfg flatlandModeConfig, overrides flatlandOverrides) (flatlandModeConfig, error) {
	if overrides.hasScannerProfile {
		cfg.scannerProfile = overrides.scannerProfile
	}
	if overrides.hasScannerSpread {
		cfg.scannerSpread = overrides.scannerSpread
	}
	if overrides.hasScannerOffset {
		cfg.scannerOffset = overrides.scannerOffset
	}
	if overrides.hasRandomizeLayout {
		cfg.randomizeLayout = overrides.randomizeLayout
	}
	if overrides.hasLayoutVariants {
		cfg.layoutVariants = overrides.layoutVariants
	}
	if overrides.hasForcedLayout {
		cfg.hasForcedLayout = true
		cfg.forcedLayout = overrides.forcedLayout
	}
	if cfg.layoutVariants <= 0 {
		cfg.layoutVariants = 1
	}
	return cfg, nil
}
