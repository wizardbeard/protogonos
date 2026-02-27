package io

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"protogonos/internal/scapeid"
)

const (
	SupportedSchemaVersion = 1
	SupportedCodecVersion  = 1
)

var (
	ErrSensorExists     = errors.New("sensor already registered")
	ErrSensorNotFound   = errors.New("sensor not found")
	ErrActuatorExists   = errors.New("actuator already registered")
	ErrActuatorNotFound = errors.New("actuator not found")
	ErrVersionMismatch  = errors.New("registry version mismatch")
	ErrIncompatible     = errors.New("component incompatible with scape")
)

type CompatibilityFn func(scape string) error

type SensorFactory func() Sensor

type ActuatorFactory func() Actuator

type SensorSpec struct {
	Name          string
	Factory       SensorFactory
	SchemaVersion int
	CodecVersion  int
	Compatible    CompatibilityFn
}

type ActuatorSpec struct {
	Name          string
	Factory       ActuatorFactory
	SchemaVersion int
	CodecVersion  int
	Compatible    CompatibilityFn
}

type registeredSensor struct {
	factory       SensorFactory
	schemaVersion int
	codecVersion  int
	compatible    CompatibilityFn
}

type registeredActuator struct {
	factory       ActuatorFactory
	schemaVersion int
	codecVersion  int
	compatible    CompatibilityFn
}

var sensorRegistry = struct {
	mu sync.RWMutex
	m  map[string]registeredSensor
}{
	m: make(map[string]registeredSensor),
}

var actuatorRegistry = struct {
	mu sync.RWMutex
	m  map[string]registeredActuator
}{
	m: make(map[string]registeredActuator),
}

func RegisterSensor(name string, factory SensorFactory) error {
	return RegisterSensorWithSpec(SensorSpec{
		Name:          name,
		Factory:       factory,
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
	})
}

func RegisterSensorWithSpec(spec SensorSpec) error {
	if spec.Name == "" {
		return errors.New("sensor name is required")
	}
	if spec.Factory == nil {
		return errors.New("sensor factory is required")
	}
	if spec.SchemaVersion != SupportedSchemaVersion || spec.CodecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: schema=%d codec=%d", ErrVersionMismatch, spec.SchemaVersion, spec.CodecVersion)
	}

	sensorRegistry.mu.Lock()
	defer sensorRegistry.mu.Unlock()

	if _, exists := sensorRegistry.m[spec.Name]; exists {
		return fmt.Errorf("%w: %s", ErrSensorExists, spec.Name)
	}
	sensorRegistry.m[spec.Name] = registeredSensor{
		factory:       spec.Factory,
		schemaVersion: spec.SchemaVersion,
		codecVersion:  spec.CodecVersion,
		compatible:    spec.Compatible,
	}
	return nil
}

func ResolveSensor(name, scape string) (Sensor, error) {
	sensorRegistry.mu.RLock()
	entry, ok := sensorRegistry.m[name]
	sensorRegistry.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrSensorNotFound, name)
	}
	if err := sensorCompatibilityError(name, entry, scapeid.Normalize(scape)); err != nil {
		return nil, err
	}
	return entry.factory(), nil
}

func SensorCompatibleWithScape(name, scape string) bool {
	sensorRegistry.mu.RLock()
	entry, ok := sensorRegistry.m[name]
	sensorRegistry.mu.RUnlock()
	if !ok {
		return false
	}
	return sensorCompatibilityError(name, entry, scapeid.Normalize(scape)) == nil
}

func ListSensorsForScape(scape string) []string {
	normalized := scapeid.Normalize(scape)

	sensorRegistry.mu.RLock()
	defer sensorRegistry.mu.RUnlock()

	names := make([]string, 0, len(sensorRegistry.m))
	for name, entry := range sensorRegistry.m {
		if sensorCompatibilityError(name, entry, normalized) != nil {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func ListSensors() []string {
	sensorRegistry.mu.RLock()
	defer sensorRegistry.mu.RUnlock()

	names := make([]string, 0, len(sensorRegistry.m))
	for n := range sensorRegistry.m {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func sensorCompatibilityError(name string, entry registeredSensor, scape string) error {
	if entry.schemaVersion != SupportedSchemaVersion || entry.codecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: %s", ErrVersionMismatch, name)
	}
	if entry.compatible != nil {
		if err := entry.compatible(scape); err != nil {
			return fmt.Errorf("%w: sensor=%s: %v", ErrIncompatible, name, err)
		}
	}
	return nil
}

func RegisterActuator(name string, factory ActuatorFactory) error {
	return RegisterActuatorWithSpec(ActuatorSpec{
		Name:          name,
		Factory:       factory,
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
	})
}

func RegisterActuatorWithSpec(spec ActuatorSpec) error {
	if spec.Name == "" {
		return errors.New("actuator name is required")
	}
	if spec.Factory == nil {
		return errors.New("actuator factory is required")
	}
	if spec.SchemaVersion != SupportedSchemaVersion || spec.CodecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: schema=%d codec=%d", ErrVersionMismatch, spec.SchemaVersion, spec.CodecVersion)
	}

	actuatorRegistry.mu.Lock()
	defer actuatorRegistry.mu.Unlock()

	if _, exists := actuatorRegistry.m[spec.Name]; exists {
		return fmt.Errorf("%w: %s", ErrActuatorExists, spec.Name)
	}
	actuatorRegistry.m[spec.Name] = registeredActuator{
		factory:       spec.Factory,
		schemaVersion: spec.SchemaVersion,
		codecVersion:  spec.CodecVersion,
		compatible:    spec.Compatible,
	}
	return nil
}

func ResolveActuator(name, scape string) (Actuator, error) {
	entry, resolvedName, ok := findRegisteredActuator(name)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrActuatorNotFound, name)
	}
	if err := actuatorCompatibilityError(resolvedName, entry, scapeid.Normalize(scape)); err != nil {
		return nil, err
	}
	return entry.factory(), nil
}

func ActuatorCompatibleWithScape(name, scape string) bool {
	entry, resolvedName, ok := findRegisteredActuator(name)
	if !ok {
		return false
	}
	return actuatorCompatibilityError(resolvedName, entry, scapeid.Normalize(scape)) == nil
}

func ListActuatorsForScape(scape string) []string {
	normalized := scapeid.Normalize(scape)

	actuatorRegistry.mu.RLock()
	defer actuatorRegistry.mu.RUnlock()

	names := make([]string, 0, len(actuatorRegistry.m))
	for name, entry := range actuatorRegistry.m {
		if actuatorCompatibilityError(name, entry, normalized) != nil {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func ListActuators() []string {
	actuatorRegistry.mu.RLock()
	defer actuatorRegistry.mu.RUnlock()

	names := make([]string, 0, len(actuatorRegistry.m))
	for n := range actuatorRegistry.m {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func actuatorCompatibilityError(name string, entry registeredActuator, scape string) error {
	if entry.schemaVersion != SupportedSchemaVersion || entry.codecVersion != SupportedCodecVersion {
		return fmt.Errorf("%w: %s", ErrVersionMismatch, name)
	}
	if entry.compatible != nil {
		if err := entry.compatible(scape); err != nil {
			return fmt.Errorf("%w: actuator=%s: %v", ErrIncompatible, name, err)
		}
	}
	return nil
}

func findRegisteredActuator(name string) (registeredActuator, string, bool) {
	lookupName := strings.TrimSpace(name)
	if lookupName == "" {
		return registeredActuator{}, "", false
	}

	actuatorRegistry.mu.RLock()
	defer actuatorRegistry.mu.RUnlock()

	if entry, ok := actuatorRegistry.m[lookupName]; ok {
		return entry, lookupName, true
	}

	canonicalName := CanonicalActuatorName(lookupName)
	if canonicalName != "" && canonicalName != lookupName {
		if entry, ok := actuatorRegistry.m[canonicalName]; ok {
			return entry, canonicalName, true
		}
	}
	return registeredActuator{}, "", false
}

func resetRegistriesForTests() {
	sensorRegistry.mu.Lock()
	sensorRegistry.m = make(map[string]registeredSensor)
	sensorRegistry.mu.Unlock()

	actuatorRegistry.mu.Lock()
	actuatorRegistry.m = make(map[string]registeredActuator)
	actuatorRegistry.mu.Unlock()

	initializeDefaultComponents()
}
