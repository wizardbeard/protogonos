package scape

import (
	"context"
	"reflect"
	"testing"
)

func TestDirectSessionProcessLifecycleReasonConsistency(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "xor",
			run: func(t *testing.T) {
				process := NewXORProcess()
				start := process.Call(ctx, XORStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, XORStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, XORRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "regression-mimic",
			run: func(t *testing.T) {
				process := NewRegressionMimicProcess()
				start := process.Call(ctx, RegressionMimicStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, RegressionMimicStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, RegressionMimicRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "cart-pole-lite",
			run: func(t *testing.T) {
				process := NewCartPoleLiteProcess()
				start := process.Call(ctx, CartPoleLiteStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, CartPoleLiteStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, CartPoleLiteRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "pole2",
			run: func(t *testing.T) {
				process := NewPole2Process()
				start := process.Call(ctx, Pole2StartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, Pole2StopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, Pole2RestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "dtm",
			run: func(t *testing.T) {
				process := NewDTMProcess()
				start := process.Call(ctx, DTMStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, DTMStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, DTMRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "gtsa",
			run: func(t *testing.T) {
				process := NewGTSAProcess()
				start := process.Call(ctx, GTSAStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, GTSAStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, GTSARestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "fx",
			run: func(t *testing.T) {
				process := NewFXProcess()
				start := process.Call(ctx, FXStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, FXStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, FXRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "epitopes",
			run: func(t *testing.T) {
				process := NewEpitopesProcess()
				start := process.Call(ctx, EpitopesStartMessage{
					OpMode: "gt",
					Params: EpitopesSimParameters{
						TableName:  "abc_pred16",
						StartIndex: 1,
						EndIndex:   2,
					},
				})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, EpitopesStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, EpitopesRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
		{
			name: "llvm-phase-ordering",
			run: func(t *testing.T) {
				process := NewLLVMPhaseOrderingProcess()
				start := process.Call(ctx, LLVMPhaseOrderingStartMessage{Mode: "validation"})
				requireResponseOK(t, start.OK, start.Err)
				stop := process.Call(ctx, LLVMPhaseOrderingStopMessage{Reason: "audit-stop"})
				requireResponseOK(t, stop.OK, stop.Err)
				requireTerminationReason(t, stop.State, "audit-stop")
				restart := process.Call(ctx, LLVMPhaseOrderingRestartMessage{})
				requireResponseOK(t, restart.OK, restart.Err)
				requireNoTerminationReason(t, restart.State)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, tc.run)
	}
}

func requireResponseOK(t *testing.T, ok bool, err error) {
	t.Helper()
	if err != nil || !ok {
		t.Fatalf("response not ok: ok=%v err=%v", ok, err)
	}
}

func requireTerminationReason(t *testing.T, state any, want string) {
	t.Helper()
	got, ok := terminationReasonField(state)
	if !ok {
		t.Fatalf("state %T does not expose TerminationReason", state)
	}
	if got != want {
		t.Fatalf("expected termination reason %q, got %q in %+v", want, got, state)
	}
}

func requireNoTerminationReason(t *testing.T, state any) {
	t.Helper()
	got, ok := terminationReasonField(state)
	if !ok {
		t.Fatalf("state %T does not expose TerminationReason", state)
	}
	if got != "" {
		t.Fatalf("expected cleared termination reason, got %q in %+v", got, state)
	}
}

func terminationReasonField(state any) (string, bool) {
	v := reflect.ValueOf(state)
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return "", false
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return "", false
	}
	field := v.FieldByName("TerminationReason")
	if !field.IsValid() || field.Kind() != reflect.String {
		return "", false
	}
	return field.String(), true
}
