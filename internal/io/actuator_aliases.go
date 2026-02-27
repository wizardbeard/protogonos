package io

import "strings"

const (
	XORSendOutputActuatorAliasName           = "xor_SendOutput"
	PBSendOutputActuatorAliasName            = "pb_SendOutput"
	DTMSendOutputActuatorAliasName           = "dtm_SendOutput"
	TwoWheelsActuatorAliasName               = "two_wheels"
	FXTradeActuatorAliasName                 = "fx_Trade"
	ABCPredActuatorAliasName                 = "abc_pred"
	GeneralPredictorActuatorAliasName        = "general_predictor"
	ChooseOptimizationPhaseActuatorAliasName = "choose_OptimizationPhase"
)

var actuatorAliasToCanonical = map[string]string{
	strings.ToLower(XORSendOutputActuatorAliasName):           XOROutputActuatorName,
	strings.ToLower(PBSendOutputActuatorAliasName):            Pole2PushActuatorName,
	strings.ToLower(DTMSendOutputActuatorAliasName):           DTMMoveActuatorName,
	strings.ToLower(TwoWheelsActuatorAliasName):               FlatlandTwoWheelsActuatorName,
	strings.ToLower(FXTradeActuatorAliasName):                 FXTradeActuatorName,
	strings.ToLower(ABCPredActuatorAliasName):                 EpitopesResponseActuatorName,
	strings.ToLower(GeneralPredictorActuatorAliasName):        GTSAPredictActuatorName,
	strings.ToLower(ChooseOptimizationPhaseActuatorAliasName): LLVMPhaseActuatorName,
}

func CanonicalActuatorName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}
	if canonical, ok := actuatorAliasToCanonical[strings.ToLower(trimmed)]; ok {
		return canonical
	}
	return trimmed
}
