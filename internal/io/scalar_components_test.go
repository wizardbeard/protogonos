package io

import (
	"context"
	"testing"
)

func TestScalarInputSensor(t *testing.T) {
	s := NewScalarInputSensor(0.25)
	values, err := s.Read(context.Background())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(values) != 1 || values[0] != 0.25 {
		t.Fatalf("unexpected sensor values: %+v", values)
	}

	s.Set(0.75)
	values, err = s.Read(context.Background())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if values[0] != 0.75 {
		t.Fatalf("unexpected updated value: %f", values[0])
	}
}

func TestScalarOutputActuator(t *testing.T) {
	a := NewScalarOutputActuator()
	if err := a.Write(context.Background(), []float64{0.9}); err != nil {
		t.Fatalf("write: %v", err)
	}
	last := a.Last()
	if len(last) != 1 || last[0] != 0.9 {
		t.Fatalf("unexpected actuator last output: %+v", last)
	}
}

func TestScalarComponentsRegistered(t *testing.T) {
	sensor, err := ResolveSensor(ScalarInputSensorName, "regression-mimic")
	if err != nil {
		t.Fatalf("resolve sensor: %v", err)
	}
	if sensor.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected sensor name: %s", sensor.Name())
	}

	actuator, err := ResolveActuator(ScalarOutputActuatorName, "regression-mimic")
	if err != nil {
		t.Fatalf("resolve actuator: %v", err)
	}
	if actuator.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected actuator name: %s", actuator.Name())
	}

	xorLeft, err := ResolveSensor(XORInputLeftSensorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor left sensor: %v", err)
	}
	if xorLeft.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected xor left sensor name: %s", xorLeft.Name())
	}

	xorRight, err := ResolveSensor(XORInputRightSensorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor right sensor: %v", err)
	}
	if xorRight.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected xor right sensor name: %s", xorRight.Name())
	}

	xorActuator, err := ResolveActuator(XOROutputActuatorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor actuator: %v", err)
	}
	if xorActuator.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected xor actuator name: %s", xorActuator.Name())
	}

	cartPosition, err := ResolveSensor(CartPolePositionSensorName, "cart-pole-lite")
	if err != nil {
		t.Fatalf("resolve cart-pole position sensor: %v", err)
	}
	if cartPosition.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected cart-pole position sensor name: %s", cartPosition.Name())
	}

	cartVelocity, err := ResolveSensor(CartPoleVelocitySensorName, "cart-pole-lite")
	if err != nil {
		t.Fatalf("resolve cart-pole velocity sensor: %v", err)
	}
	if cartVelocity.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected cart-pole velocity sensor name: %s", cartVelocity.Name())
	}

	cartForce, err := ResolveActuator(CartPoleForceActuatorName, "cart-pole-lite")
	if err != nil {
		t.Fatalf("resolve cart-pole force actuator: %v", err)
	}
	if cartForce.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected cart-pole force actuator name: %s", cartForce.Name())
	}

	flatDistance, err := ResolveSensor(FlatlandDistanceSensorName, "flatland")
	if err != nil {
		t.Fatalf("resolve flatland distance sensor: %v", err)
	}
	if flatDistance.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected flatland distance sensor name: %s", flatDistance.Name())
	}

	flatEnergy, err := ResolveSensor(FlatlandEnergySensorName, "flatland")
	if err != nil {
		t.Fatalf("resolve flatland energy sensor: %v", err)
	}
	if flatEnergy.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected flatland energy sensor name: %s", flatEnergy.Name())
	}

	flatMove, err := ResolveActuator(FlatlandMoveActuatorName, "flatland")
	if err != nil {
		t.Fatalf("resolve flatland move actuator: %v", err)
	}
	if flatMove.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected flatland move actuator name: %s", flatMove.Name())
	}

	gtsaInput, err := ResolveSensor(GTSAInputSensorName, "gtsa")
	if err != nil {
		t.Fatalf("resolve gtsa input sensor: %v", err)
	}
	if gtsaInput.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected gtsa input sensor name: %s", gtsaInput.Name())
	}

	gtsaPredict, err := ResolveActuator(GTSAPredictActuatorName, "gtsa")
	if err != nil {
		t.Fatalf("resolve gtsa predict actuator: %v", err)
	}
	if gtsaPredict.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected gtsa predict actuator name: %s", gtsaPredict.Name())
	}

	fxPrice, err := ResolveSensor(FXPriceSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx price sensor: %v", err)
	}
	if fxPrice.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx price sensor name: %s", fxPrice.Name())
	}

	fxSignal, err := ResolveSensor(FXSignalSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx signal sensor: %v", err)
	}
	if fxSignal.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx signal sensor name: %s", fxSignal.Name())
	}

	fxTrade, err := ResolveActuator(FXTradeActuatorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx trade actuator: %v", err)
	}
	if fxTrade.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected fx trade actuator name: %s", fxTrade.Name())
	}
}
