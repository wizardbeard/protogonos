package io

import (
	"context"
	"fmt"
	"sync"
)

const (
	ScalarInputSensorName             = "scalar_input"
	ScalarOutputActuatorName          = "scalar_output"
	XORInputLeftSensorName            = "xor_input_left"
	XORInputRightSensorName           = "xor_input_right"
	XOROutputActuatorName             = "xor_output"
	CartPolePositionSensorName        = "cart_pole_position"
	CartPoleVelocitySensorName        = "cart_pole_velocity"
	CartPoleForceActuatorName         = "cart_pole_force"
	Pole2CartPositionSensorName       = "pole2_cart_position"
	Pole2CartVelocitySensorName       = "pole2_cart_velocity"
	Pole2Angle1SensorName             = "pole2_angle_1"
	Pole2Velocity1SensorName          = "pole2_velocity_1"
	Pole2Angle2SensorName             = "pole2_angle_2"
	Pole2Velocity2SensorName          = "pole2_velocity_2"
	Pole2RunProgressSensorName        = "pole2_run_progress"
	Pole2StepProgressSensorName       = "pole2_step_progress"
	Pole2FitnessSignalSensorName      = "pole2_fitness_signal"
	Pole2PushActuatorName             = "pole2_push"
	FlatlandDistanceSensorName        = "flatland_distance"
	FlatlandEnergySensorName          = "flatland_energy"
	FlatlandPoisonSensorName          = "flatland_poison"
	FlatlandWallSensorName            = "flatland_wall"
	FlatlandFoodProximitySensorName   = "flatland_food_proximity"
	FlatlandPoisonProximitySensorName = "flatland_poison_proximity"
	FlatlandWallProximitySensorName   = "flatland_wall_proximity"
	FlatlandResourceBalanceSensorName = "flatland_resource_balance"
	FlatlandMoveActuatorName          = "flatland_move"
	DTMRangeLeftSensorName            = "dtm_range_left"
	DTMRangeFrontSensorName           = "dtm_range_front"
	DTMRangeRightSensorName           = "dtm_range_right"
	DTMRewardSensorName               = "dtm_reward"
	DTMRunProgressSensorName          = "dtm_run_progress"
	DTMStepProgressSensorName         = "dtm_step_progress"
	DTMSwitchedSensorName             = "dtm_switched"
	DTMMoveActuatorName               = "dtm_move"
	GTSAInputSensorName               = "gtsa_input"
	GTSADeltaSensorName               = "gtsa_delta"
	GTSAWindowMeanSensorName          = "gtsa_window_mean"
	GTSAProgressSensorName            = "gtsa_progress"
	GTSAPredictActuatorName           = "gtsa_predict"
	FXPriceSensorName                 = "fx_price"
	FXSignalSensorName                = "fx_signal"
	FXMomentumSensorName              = "fx_momentum"
	FXVolatilitySensorName            = "fx_volatility"
	FXNAVSensorName                   = "fx_nav"
	FXDrawdownSensorName              = "fx_drawdown"
	FXPositionSensorName              = "fx_position"
	FXEntrySensorName                 = "fx_entry"
	FXPercentChangeSensorName         = "fx_percent_change"
	FXPrevPercentChangeSensorName     = "fx_prev_percentage_change"
	FXProfitSensorName                = "fx_profit"
	FXTradeActuatorName               = "fx_trade"
	EpitopesSignalSensorName          = "epitopes_signal"
	EpitopesMemorySensorName          = "epitopes_memory"
	EpitopesTargetSensorName          = "epitopes_target"
	EpitopesProgressSensorName        = "epitopes_progress"
	EpitopesMarginSensorName          = "epitopes_margin"
	EpitopesResponseActuatorName      = "epitopes_response"
	LLVMComplexitySensorName          = "llvm_complexity"
	LLVMPassIndexSensorName           = "llvm_pass_index"
	LLVMAlignmentSensorName           = "llvm_alignment"
	LLVMDiversitySensorName           = "llvm_diversity"
	LLVMRuntimeGainSensorName         = "llvm_runtime_gain"
	LLVMPhaseActuatorName             = "llvm_phase"
)

type ScalarInputSensor struct {
	mu    sync.RWMutex
	value float64
}

func NewScalarInputSensor(initial float64) *ScalarInputSensor {
	return &ScalarInputSensor{value: initial}
}

func (s *ScalarInputSensor) Name() string {
	return ScalarInputSensorName
}

func (s *ScalarInputSensor) Read(_ context.Context) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return []float64{s.value}, nil
}

func (s *ScalarInputSensor) Set(value float64) {
	s.mu.Lock()
	s.value = value
	s.mu.Unlock()
}

type ScalarOutputActuator struct {
	mu   sync.RWMutex
	last []float64
}

func NewScalarOutputActuator() *ScalarOutputActuator {
	return &ScalarOutputActuator{}
}

func (a *ScalarOutputActuator) Name() string {
	return ScalarOutputActuatorName
}

func (a *ScalarOutputActuator) Write(_ context.Context, values []float64) error {
	a.mu.Lock()
	a.last = append([]float64(nil), values...)
	a.mu.Unlock()
	return nil
}

func (a *ScalarOutputActuator) Last() []float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]float64(nil), a.last...)
}

func init() {
	initializeDefaultComponents()
}

func initializeDefaultComponents() {
	err := RegisterSensorWithSpec(SensorSpec{
		Name:          ScalarInputSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "regression-mimic" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          CartPolePositionSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          CartPoleVelocitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2CartPositionSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2CartVelocitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2Angle1SensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2Velocity1SensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2Angle2SensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2Velocity2SensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2RunProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2StepProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          Pole2FitnessSignalSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          XORInputLeftSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          XORInputRightSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandDistanceSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandEnergySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandPoisonSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandWallSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandFoodProximitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandPoisonProximitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandWallProximitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FlatlandResourceBalanceSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMRangeLeftSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMRangeFrontSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMRangeRightSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMRewardSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMRunProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMStepProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          DTMSwitchedSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          GTSAInputSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "gtsa" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          GTSADeltaSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "gtsa" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          GTSAWindowMeanSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "gtsa" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          GTSAProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "gtsa" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXPriceSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXSignalSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXMomentumSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXVolatilitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXNAVSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXDrawdownSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXPositionSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXEntrySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXPercentChangeSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXPrevPercentChangeSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          FXProfitSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          EpitopesSignalSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          EpitopesMemorySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          EpitopesTargetSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          EpitopesProgressSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          EpitopesMarginSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          LLVMComplexitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          LLVMPassIndexSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          LLVMAlignmentSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          LLVMDiversitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          LLVMRuntimeGainSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}

	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          ScalarOutputActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "regression-mimic" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          XOROutputActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          CartPoleForceActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          Pole2PushActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "pole2-balancing" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          FlatlandMoveActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "flatland" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          DTMMoveActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "dtm" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          GTSAPredictActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "gtsa" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          FXTradeActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "fx" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          EpitopesResponseActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "epitopes" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          LLVMPhaseActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "llvm-phase-ordering" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
}
