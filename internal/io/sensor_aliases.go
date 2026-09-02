package io

import "strings"

const (
	XORGetInputSensorAliasName       = "xor_GetInput"
	PBGetInputSensorAliasName        = "pb_GetInput"
	DTMGetInputSensorAliasName       = "dtm_GetInput"
	DistanceScannerSensorAliasName   = "distance_scanner"
	ColorScannerSensorAliasName      = "color_scanner"
	EnergyScannerSensorAliasName     = "energy_scaner"
	EnergyScannerSensorCorrectedName = "energy_scanner"
	FXPCISensorAliasName             = "fx_PCI"
	FXPLISensorAliasName             = "fx_PLI"
	FXInternalsSensorAliasName       = "fx_Internals"
	ABCPredSensorAliasName           = "abc_pred"
	GeneralPredictorSensorAliasName  = "general_predictor"
	BitCodeStatisticsSensorAliasName = "get_BitCodeStatistics"
)

var sensorAliasToCanonical = map[string]string{
	strings.ToLower(XORGetInputSensorAliasName):       XORInputLeftSensorName,
	strings.ToLower(PBGetInputSensorAliasName):        Pole2CartPositionSensorName,
	strings.ToLower(DTMGetInputSensorAliasName):       DTMRangeFrontSensorName,
	strings.ToLower(DistanceScannerSensorAliasName):   FlatlandDistanceScan0SensorName,
	strings.ToLower(ColorScannerSensorAliasName):      FlatlandColorScan0SensorName,
	strings.ToLower(EnergyScannerSensorAliasName):     FlatlandEnergyScan0SensorName,
	strings.ToLower(EnergyScannerSensorCorrectedName): FlatlandEnergyScan0SensorName,
	strings.ToLower(FXPCISensorAliasName):             FXPercentChangeSensorName,
	strings.ToLower(FXPLISensorAliasName):             FXPercentChangeSensorName,
	strings.ToLower(FXInternalsSensorAliasName):       FXNAVSensorName,
	strings.ToLower(ABCPredSensorAliasName):           EpitopesSignalSensorName,
	strings.ToLower(GeneralPredictorSensorAliasName):  GTSAInputSensorName,
	strings.ToLower(BitCodeStatisticsSensorAliasName): LLVMRuntimeGainSensorName,
}

func CanonicalSensorName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}
	if canonical, ok := sensorAliasToCanonical[strings.ToLower(trimmed)]; ok {
		return canonical
	}
	return trimmed
}
