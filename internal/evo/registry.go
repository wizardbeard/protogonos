package evo

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"protogonos/internal/model"
)

const (
	SupportedSchemaVersion = 1
	SupportedCodecVersion  = 1
)

var (
	ErrOperatorExists       = errors.New("operator already registered")
	ErrOperatorNotFound     = errors.New("operator not found")
	ErrOperatorIncompatible = errors.New("operator incompatible with genome")
	ErrVersionMismatch      = errors.New("operator version mismatch")
)

type CompatibilityFn func(genome model.Genome) error

type OperatorSpec struct {
	Name          string
	Operator      Operator
	SchemaVersion int
	CodecVersion  int
	Compatible    CompatibilityFn
}

type registeredOperator struct {
	operator      Operator
	schemaVersion int
	codecVersion  int
	compatible    CompatibilityFn
}

var operatorRegistry = struct {
	mu sync.RWMutex
	m  map[string]registeredOperator
}{
	m: make(map[string]registeredOperator),
}

// RegisterOperator registers an operator with default schema and codec versions.
func RegisterOperator(name string, op Operator) error {
	return RegisterOperatorWithSpec(OperatorSpec{
		Name:          name,
		Operator:      op,
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
	})
}

// RegisterOperatorWithSpec registers an operator with explicit versioning and compatibility metadata.
func RegisterOperatorWithSpec(spec OperatorSpec) error {
	if spec.Name == "" {
		return errors.New("operator name is required")
	}
	if spec.Operator == nil {
		return errors.New("operator is required")
	}
	if spec.SchemaVersion != SupportedSchemaVersion || spec.CodecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: schema=%d codec=%d", ErrVersionMismatch, spec.SchemaVersion, spec.CodecVersion)
	}

	operatorRegistry.mu.Lock()
	defer operatorRegistry.mu.Unlock()

	if _, exists := operatorRegistry.m[spec.Name]; exists {
		return fmt.Errorf("%w: %s", ErrOperatorExists, spec.Name)
	}

	operatorRegistry.m[spec.Name] = registeredOperator{
		operator:      spec.Operator,
		schemaVersion: spec.SchemaVersion,
		codecVersion:  spec.CodecVersion,
		compatible:    spec.Compatible,
	}
	return nil
}

// ResolveOperator returns a registered operator only if record versions and compatibility checks pass.
func ResolveOperator(name string, genome model.Genome) (Operator, error) {
	operatorRegistry.mu.RLock()
	entry, ok := operatorRegistry.m[name]
	operatorRegistry.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrOperatorNotFound, name)
	}
	if genome.SchemaVersion != entry.schemaVersion || genome.CodecVersion != entry.codecVersion {
		return nil, fmt.Errorf("%w: operator=%s expected(schema=%d codec=%d) got(schema=%d codec=%d)",
			ErrVersionMismatch,
			name,
			entry.schemaVersion,
			entry.codecVersion,
			genome.SchemaVersion,
			genome.CodecVersion,
		)
	}
	if entry.compatible != nil {
		if err := entry.compatible(genome); err != nil {
			return nil, fmt.Errorf("%w: %s: %v", ErrOperatorIncompatible, name, err)
		}
	}
	return entry.operator, nil
}

func ListOperators() []string {
	operatorRegistry.mu.RLock()
	defer operatorRegistry.mu.RUnlock()

	names := make([]string, 0, len(operatorRegistry.m))
	for name := range operatorRegistry.m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func resetOperatorRegistryForTests() {
	operatorRegistry.mu.Lock()
	defer operatorRegistry.mu.Unlock()
	operatorRegistry.m = make(map[string]registeredOperator)
}
