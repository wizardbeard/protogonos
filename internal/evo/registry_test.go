package evo

import (
	"context"
	"errors"
	"testing"

	"protogonos/internal/model"
)

type noopOperator struct{}

func (noopOperator) Name() string { return "noop" }

func (noopOperator) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	return genome, nil
}

func TestRegisterAndResolveOperator(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	if err := RegisterOperator("noop", noopOperator{}); err != nil {
		t.Fatalf("register: %v", err)
	}

	op, err := ResolveOperator("noop", model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: SupportedSchemaVersion, CodecVersion: SupportedCodecVersion},
	})
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if op.Name() != "noop" {
		t.Fatalf("unexpected operator: %s", op.Name())
	}
}

func TestRegisterOperatorDuplicate(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	if err := RegisterOperator("noop", noopOperator{}); err != nil {
		t.Fatalf("first register: %v", err)
	}
	if err := RegisterOperator("noop", noopOperator{}); !errors.Is(err, ErrOperatorExists) {
		t.Fatalf("expected ErrOperatorExists, got: %v", err)
	}
}

func TestRegisterOperatorValidation(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	if err := RegisterOperator("", noopOperator{}); err == nil {
		t.Fatal("expected empty name error")
	}
	if err := RegisterOperator("nil", nil); err == nil {
		t.Fatal("expected nil operator error")
	}
	if err := RegisterOperatorWithSpec(OperatorSpec{
		Name:          "bad-version",
		Operator:      noopOperator{},
		SchemaVersion: 99,
		CodecVersion:  1,
	}); !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestResolveOperatorNotFound(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	_, err := ResolveOperator("missing", model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: SupportedSchemaVersion, CodecVersion: SupportedCodecVersion},
	})
	if !errors.Is(err, ErrOperatorNotFound) {
		t.Fatalf("expected ErrOperatorNotFound, got: %v", err)
	}
}

func TestResolveOperatorVersionMismatch(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	if err := RegisterOperator("noop", noopOperator{}); err != nil {
		t.Fatalf("register: %v", err)
	}

	_, err := ResolveOperator("noop", model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: SupportedSchemaVersion, CodecVersion: SupportedCodecVersion + 1},
	})
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestResolveOperatorCompatibility(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	compatibilityErr := errors.New("requires at least one synapse")
	if err := RegisterOperatorWithSpec(OperatorSpec{
		Name:          "needs-synapse",
		Operator:      noopOperator{},
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(genome model.Genome) error {
			if len(genome.Synapses) == 0 {
				return compatibilityErr
			}
			return nil
		},
	}); err != nil {
		t.Fatalf("register with compatibility: %v", err)
	}

	_, err := ResolveOperator("needs-synapse", model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: SupportedSchemaVersion, CodecVersion: SupportedCodecVersion},
	})
	if !errors.Is(err, ErrOperatorIncompatible) {
		t.Fatalf("expected ErrOperatorIncompatible, got: %v", err)
	}
}

func TestListOperatorsSorted(t *testing.T) {
	resetOperatorRegistryForTests()
	t.Cleanup(resetOperatorRegistryForTests)

	if err := RegisterOperator("b-op", noopOperator{}); err != nil {
		t.Fatalf("register b-op: %v", err)
	}
	if err := RegisterOperator("a-op", noopOperator{}); err != nil {
		t.Fatalf("register a-op: %v", err)
	}

	names := ListOperators()
	if len(names) != 2 || names[0] != "a-op" || names[1] != "b-op" {
		t.Fatalf("unexpected operator list: %+v", names)
	}
}
