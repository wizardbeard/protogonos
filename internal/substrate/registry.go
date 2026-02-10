package substrate

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

var (
	ErrCPPExists   = errors.New("cpp already registered")
	ErrCPPNotFound = errors.New("cpp not found")
	ErrCEPExists   = errors.New("cep already registered")
	ErrCEPNotFound = errors.New("cep not found")
)

type CPPFactory func() CPP
type CEPFactory func() CEP

var cppRegistry = struct {
	mu sync.RWMutex
	m  map[string]CPPFactory
}{
	m: make(map[string]CPPFactory),
}

var cepRegistry = struct {
	mu sync.RWMutex
	m  map[string]CEPFactory
}{
	m: make(map[string]CEPFactory),
}

func RegisterCPP(name string, factory CPPFactory) error {
	if name == "" {
		return errors.New("cpp name is required")
	}
	if factory == nil {
		return errors.New("cpp factory is required")
	}

	cppRegistry.mu.Lock()
	defer cppRegistry.mu.Unlock()
	if _, exists := cppRegistry.m[name]; exists {
		return fmt.Errorf("%w: %s", ErrCPPExists, name)
	}
	cppRegistry.m[name] = factory
	return nil
}

func ResolveCPP(name string) (CPP, error) {
	cppRegistry.mu.RLock()
	factory, ok := cppRegistry.m[name]
	cppRegistry.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrCPPNotFound, name)
	}
	return factory(), nil
}

func ListCPPs() []string {
	cppRegistry.mu.RLock()
	defer cppRegistry.mu.RUnlock()
	names := make([]string, 0, len(cppRegistry.m))
	for name := range cppRegistry.m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func RegisterCEP(name string, factory CEPFactory) error {
	if name == "" {
		return errors.New("cep name is required")
	}
	if factory == nil {
		return errors.New("cep factory is required")
	}

	cepRegistry.mu.Lock()
	defer cepRegistry.mu.Unlock()
	if _, exists := cepRegistry.m[name]; exists {
		return fmt.Errorf("%w: %s", ErrCEPExists, name)
	}
	cepRegistry.m[name] = factory
	return nil
}

func ResolveCEP(name string) (CEP, error) {
	cepRegistry.mu.RLock()
	factory, ok := cepRegistry.m[name]
	cepRegistry.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrCEPNotFound, name)
	}
	return factory(), nil
}

func ListCEPs() []string {
	cepRegistry.mu.RLock()
	defer cepRegistry.mu.RUnlock()
	names := make([]string, 0, len(cepRegistry.m))
	for name := range cepRegistry.m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func resetRegistriesForTests() {
	cppRegistry.mu.Lock()
	cppRegistry.m = make(map[string]CPPFactory)
	cppRegistry.mu.Unlock()

	cepRegistry.mu.Lock()
	cepRegistry.m = make(map[string]CEPFactory)
	cepRegistry.mu.Unlock()

	initializeDefaultComponents()
}
