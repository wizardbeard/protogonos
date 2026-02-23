package morphology

import protoio "protogonos/internal/io"

type LLVMPhaseOrderingMorphology struct{}

func (LLVMPhaseOrderingMorphology) Name() string {
	return "llvm-phase-ordering-v1"
}

func (LLVMPhaseOrderingMorphology) Sensors() []string {
	return []string{
		protoio.LLVMComplexitySensorName,
		protoio.LLVMPassIndexSensorName,
	}
}

func (LLVMPhaseOrderingMorphology) Actuators() []string {
	return []string{protoio.LLVMPhaseActuatorName}
}

func (LLVMPhaseOrderingMorphology) Compatible(scape string) bool {
	return scape == "llvm-phase-ordering"
}
