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
	pole2CPos, err := ResolveSensor(Pole2CartPositionSensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 cart position sensor: %v", err)
	}
	if pole2CPos.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 cart position sensor name: %s", pole2CPos.Name())
	}
	pole2CVel, err := ResolveSensor(Pole2CartVelocitySensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 cart velocity sensor: %v", err)
	}
	if pole2CVel.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 cart velocity sensor name: %s", pole2CVel.Name())
	}
	pole2Angle1, err := ResolveSensor(Pole2Angle1SensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 angle1 sensor: %v", err)
	}
	if pole2Angle1.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 angle1 sensor name: %s", pole2Angle1.Name())
	}
	pole2Velocity1, err := ResolveSensor(Pole2Velocity1SensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 velocity1 sensor: %v", err)
	}
	if pole2Velocity1.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 velocity1 sensor name: %s", pole2Velocity1.Name())
	}
	pole2Angle2, err := ResolveSensor(Pole2Angle2SensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 angle2 sensor: %v", err)
	}
	if pole2Angle2.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 angle2 sensor name: %s", pole2Angle2.Name())
	}
	pole2Velocity2, err := ResolveSensor(Pole2Velocity2SensorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 velocity2 sensor: %v", err)
	}
	if pole2Velocity2.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected pole2 velocity2 sensor name: %s", pole2Velocity2.Name())
	}

	pole2Push, err := ResolveActuator(Pole2PushActuatorName, "pole2-balancing")
	if err != nil {
		t.Fatalf("resolve pole2 push actuator: %v", err)
	}
	if pole2Push.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected pole2 push actuator name: %s", pole2Push.Name())
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
	dtmRangeLeft, err := ResolveSensor(DTMRangeLeftSensorName, "dtm")
	if err != nil {
		t.Fatalf("resolve dtm range-left sensor: %v", err)
	}
	if dtmRangeLeft.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected dtm range-left sensor name: %s", dtmRangeLeft.Name())
	}
	dtmRangeFront, err := ResolveSensor(DTMRangeFrontSensorName, "dtm")
	if err != nil {
		t.Fatalf("resolve dtm range-front sensor: %v", err)
	}
	if dtmRangeFront.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected dtm range-front sensor name: %s", dtmRangeFront.Name())
	}
	dtmRangeRight, err := ResolveSensor(DTMRangeRightSensorName, "dtm")
	if err != nil {
		t.Fatalf("resolve dtm range-right sensor: %v", err)
	}
	if dtmRangeRight.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected dtm range-right sensor name: %s", dtmRangeRight.Name())
	}
	dtmReward, err := ResolveSensor(DTMRewardSensorName, "dtm")
	if err != nil {
		t.Fatalf("resolve dtm reward sensor: %v", err)
	}
	if dtmReward.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected dtm reward sensor name: %s", dtmReward.Name())
	}

	dtmMove, err := ResolveActuator(DTMMoveActuatorName, "dtm")
	if err != nil {
		t.Fatalf("resolve dtm move actuator: %v", err)
	}
	if dtmMove.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected dtm move actuator name: %s", dtmMove.Name())
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
	fxMomentum, err := ResolveSensor(FXMomentumSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx momentum sensor: %v", err)
	}
	if fxMomentum.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx momentum sensor name: %s", fxMomentum.Name())
	}
	fxVolatility, err := ResolveSensor(FXVolatilitySensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx volatility sensor: %v", err)
	}
	if fxVolatility.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx volatility sensor name: %s", fxVolatility.Name())
	}
	fxNAV, err := ResolveSensor(FXNAVSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx nav sensor: %v", err)
	}
	if fxNAV.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx nav sensor name: %s", fxNAV.Name())
	}
	fxDrawdown, err := ResolveSensor(FXDrawdownSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx drawdown sensor: %v", err)
	}
	if fxDrawdown.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx drawdown sensor name: %s", fxDrawdown.Name())
	}
	fxPosition, err := ResolveSensor(FXPositionSensorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx position sensor: %v", err)
	}
	if fxPosition.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected fx position sensor name: %s", fxPosition.Name())
	}

	fxTrade, err := ResolveActuator(FXTradeActuatorName, "fx")
	if err != nil {
		t.Fatalf("resolve fx trade actuator: %v", err)
	}
	if fxTrade.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected fx trade actuator name: %s", fxTrade.Name())
	}

	epitopesSignal, err := ResolveSensor(EpitopesSignalSensorName, "epitopes")
	if err != nil {
		t.Fatalf("resolve epitopes signal sensor: %v", err)
	}
	if epitopesSignal.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected epitopes signal sensor name: %s", epitopesSignal.Name())
	}
	epitopesMemory, err := ResolveSensor(EpitopesMemorySensorName, "epitopes")
	if err != nil {
		t.Fatalf("resolve epitopes memory sensor: %v", err)
	}
	if epitopesMemory.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected epitopes memory sensor name: %s", epitopesMemory.Name())
	}

	epitopesResponse, err := ResolveActuator(EpitopesResponseActuatorName, "epitopes")
	if err != nil {
		t.Fatalf("resolve epitopes response actuator: %v", err)
	}
	if epitopesResponse.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected epitopes response actuator name: %s", epitopesResponse.Name())
	}

	llvmComplexity, err := ResolveSensor(LLVMComplexitySensorName, "llvm-phase-ordering")
	if err != nil {
		t.Fatalf("resolve llvm complexity sensor: %v", err)
	}
	if llvmComplexity.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected llvm complexity sensor name: %s", llvmComplexity.Name())
	}
	llvmPassIndex, err := ResolveSensor(LLVMPassIndexSensorName, "llvm-phase-ordering")
	if err != nil {
		t.Fatalf("resolve llvm pass-index sensor: %v", err)
	}
	if llvmPassIndex.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected llvm pass-index sensor name: %s", llvmPassIndex.Name())
	}

	llvmPhase, err := ResolveActuator(LLVMPhaseActuatorName, "llvm-phase-ordering")
	if err != nil {
		t.Fatalf("resolve llvm phase actuator: %v", err)
	}
	if llvmPhase.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected llvm phase actuator name: %s", llvmPhase.Name())
	}

	if _, err := ResolveSensor(Pole2CartPositionSensorName, "pb_sim"); err != nil {
		t.Fatalf("resolve pole2 alias sensor pb_sim: %v", err)
	}
	if _, err := ResolveSensor(DTMRangeFrontSensorName, "dtm_sim"); err != nil {
		t.Fatalf("resolve dtm alias sensor dtm_sim: %v", err)
	}
	if _, err := ResolveSensor(GTSAInputSensorName, "scape_GTSA"); err != nil {
		t.Fatalf("resolve gtsa alias sensor scape_GTSA: %v", err)
	}
	if _, err := ResolveActuator(LLVMPhaseActuatorName, "scape_LLVMPhaseOrdering"); err != nil {
		t.Fatalf("resolve llvm alias actuator scape_LLVMPhaseOrdering: %v", err)
	}
}
