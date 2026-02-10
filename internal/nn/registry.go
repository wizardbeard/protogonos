package nn

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

const (
	SupportedSchemaVersion = 1
	SupportedCodecVersion  = 1
)

var (
	ErrActivationExists   = errors.New("activation already registered")
	ErrActivationNotFound = errors.New("activation not found")
	ErrActivationVersion  = errors.New("activation version mismatch")
)

type ActivationFunc func(x float64) float64

type ActivationSpec struct {
	Name          string
	Func          ActivationFunc
	SchemaVersion int
	CodecVersion  int
}

type registeredActivation struct {
	fn            ActivationFunc
	schemaVersion int
	codecVersion  int
}

var activationRegistry = struct {
	mu sync.RWMutex
	m  map[string]registeredActivation
}{
	m: make(map[string]registeredActivation),
}

func init() {
	initializeBuiltInActivations()
}

func initializeBuiltInActivations() {
	MustRegisterActivation("identity", func(x float64) float64 { return x })
	MustRegisterActivation("relu", func(x float64) float64 {
		if x < 0 {
			return 0
		}
		return x
	})
	MustRegisterActivation("tanh", math.Tanh)
	MustRegisterActivation("sigmoid", func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	})
}

func RegisterActivation(name string, fn ActivationFunc) error {
	return RegisterActivationWithSpec(ActivationSpec{
		Name:          name,
		Func:          fn,
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
	})
}

func MustRegisterActivation(name string, fn ActivationFunc) {
	if err := RegisterActivation(name, fn); err != nil {
		panic(err)
	}
}

func RegisterActivationWithSpec(spec ActivationSpec) error {
	if spec.Name == "" {
		return errors.New("activation name is required")
	}
	if spec.Func == nil {
		return errors.New("activation function is required")
	}
	if spec.SchemaVersion != SupportedSchemaVersion || spec.CodecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: schema=%d codec=%d", ErrActivationVersion, spec.SchemaVersion, spec.CodecVersion)
	}

	activationRegistry.mu.Lock()
	defer activationRegistry.mu.Unlock()

	if _, exists := activationRegistry.m[spec.Name]; exists {
		return fmt.Errorf("%w: %s", ErrActivationExists, spec.Name)
	}

	activationRegistry.m[spec.Name] = registeredActivation{
		fn:            spec.Func,
		schemaVersion: spec.SchemaVersion,
		codecVersion:  spec.CodecVersion,
	}
	return nil
}

func GetActivation(name string) (ActivationFunc, error) {
	activationRegistry.mu.RLock()
	entry, ok := activationRegistry.m[name]
	activationRegistry.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrActivationNotFound, name)
	}
	if entry.schemaVersion != SupportedSchemaVersion || entry.codecVersion != SupportedCodecVersion {
		return nil, fmt.Errorf("%w: %s", ErrActivationVersion, name)
	}
	return entry.fn, nil
}

func ListActivations() []string {
	activationRegistry.mu.RLock()
	defer activationRegistry.mu.RUnlock()

	names := make([]string, 0, len(activationRegistry.m))
	for name := range activationRegistry.m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func resetActivationRegistryForTests() {
	activationRegistry.mu.Lock()
	activationRegistry.m = make(map[string]registeredActivation)
	activationRegistry.mu.Unlock()
	initializeBuiltInActivations()
}
