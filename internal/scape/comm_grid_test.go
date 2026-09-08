package scape

import (
	"context"
	"testing"
)

func TestCommGridSimulatorCompletesDelivery(t *testing.T) {
	sim := NewCommGridSimulator(CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: 8,
		Key:      CommGridPoint{X: 1, Y: 0},
		Goal:     CommGridPoint{X: 2, Y: 0},
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	})

	actions := []CommGridAction{CommGridEast, CommGridPick, CommGridEast, CommGridDrop}
	var result CommGridStepResult
	var err error
	for _, action := range actions {
		result, err = sim.Step(context.Background(), CommGridStepInput{
			AgentID: "agent-1",
			Action:  action,
		})
		if err != nil {
			t.Fatalf("Step(%s): %v", action, err)
		}
	}

	if !result.Completed || !result.Done {
		t.Fatalf("expected completed terminal result, got %+v", result)
	}
	if result.Fitness <= 1.0 {
		t.Fatalf("expected strong completed fitness, got %f trace=%+v", result.Fitness, result.Trace)
	}
	state, ok := sim.AgentState("agent-1")
	if !ok {
		t.Fatal("expected agent state")
	}
	if state.Carrying {
		t.Fatalf("expected key dropped at goal, state=%+v", state)
	}
}

func TestCommGridSimulatorPenalizesInvalidActionsAndMessages(t *testing.T) {
	sim := NewCommGridSimulator(CommGridConfig{
		Width:    2,
		Height:   2,
		MaxSteps: 4,
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	})

	result, err := sim.Step(context.Background(), CommGridStepInput{
		AgentID: "agent-1",
		Action:  CommGridWest,
		Message: "blocked",
		To:      "agent-2",
		Tokens:  12,
	})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if !result.InvalidAction {
		t.Fatalf("expected invalid wall move, result=%+v", result)
	}
	if got := len(sim.Messages()); got != 1 {
		t.Fatalf("expected one recorded message, got %d", got)
	}
	trace := result.Trace
	if trace["invalid_actions"] != 1 || trace["messages"] != 1 || trace["tokens"] != 12 {
		t.Fatalf("unexpected trace: %+v", trace)
	}
}

func TestCommGridScapeRewardsPolicyThatCompletesGoal(t *testing.T) {
	cfg := CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: 8,
		Key:      CommGridPoint{X: 1, Y: 0},
		Goal:     CommGridPoint{X: 2, Y: 0},
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	}
	sc := CommGridScape{Config: cfg}
	solver := scriptedStepAgent{
		id: "agent-1",
		fn: func(input []float64) []float64 {
			carrying := len(input) > 2 && input[2] > 0.5
			dx := input[0]
			if !carrying && dx == 0 {
				return []float64{0, 0, 1}
			}
			if carrying && dx == 0 {
				return []float64{0, 0, -1}
			}
			return []float64{1, 0, 0}
		},
	}
	stuck := scriptedStepAgent{
		id: "agent-1",
		fn: func([]float64) []float64 {
			return []float64{0, 0, 0}
		},
	}

	solverFitness, solverTrace, err := sc.Evaluate(context.Background(), solver)
	if err != nil {
		t.Fatalf("solver Evaluate: %v", err)
	}
	stuckFitness, stuckTrace, err := sc.Evaluate(context.Background(), stuck)
	if err != nil {
		t.Fatalf("stuck Evaluate: %v", err)
	}
	if solverFitness <= stuckFitness {
		t.Fatalf("expected solver to beat stuck policy: solver=%f stuck=%f solver_trace=%+v stuck_trace=%+v", solverFitness, stuckFitness, solverTrace, stuckTrace)
	}
	if completed, _ := solverTrace["completed"].(bool); !completed {
		t.Fatalf("expected solver trace to complete, got %+v", solverTrace)
	}
}

func TestDecodeCommGridActionValidatesOutputWidth(t *testing.T) {
	if _, err := DecodeCommGridAction([]float64{1, 0}); err == nil {
		t.Fatal("expected width error")
	}
}

func TestDecodeCommGridLanguageActionMapsStructuredMessage(t *testing.T) {
	input, err := DecodeCommGridLanguageAction(" agent-1 ", []byte(`{
		"action": "move-east",
		"message": "key is east",
		"to": "agent-2"
	}`))
	if err != nil {
		t.Fatalf("DecodeCommGridLanguageAction: %v", err)
	}
	if input.AgentID != "agent-1" || input.Action != CommGridEast || input.To != "agent-2" {
		t.Fatalf("unexpected decoded input: %+v", input)
	}
	if input.Message != "key is east" || input.Tokens != 3 {
		t.Fatalf("unexpected decoded message: %+v", input)
	}
}

func TestDecodeCommGridLanguageActionRejectsInvalidAction(t *testing.T) {
	if _, err := DecodeCommGridLanguageAction("agent-1", []byte(`{"action":"teleport"}`)); err == nil {
		t.Fatal("expected invalid action error")
	}
}

func TestCommGridSimulatorRecordsDecodedLanguageAction(t *testing.T) {
	sim := NewCommGridSimulator(CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: 8,
		Key:      CommGridPoint{X: 1, Y: 0},
		Goal:     CommGridPoint{X: 2, Y: 0},
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	})
	input, err := DecodeCommGridLanguageAction("agent-1", []byte(`{
		"action": "move_east",
		"message": "moving to key",
		"to": "all",
		"tokens": 8
	}`))
	if err != nil {
		t.Fatalf("DecodeCommGridLanguageAction: %v", err)
	}
	result, err := sim.Step(context.Background(), input)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if result.InvalidAction {
		t.Fatalf("expected decoded move to be valid, result=%+v", result)
	}
	messages := sim.Messages()
	if len(messages) != 1 || messages[0].Text != "moving to key" || messages[0].Tokens != 8 {
		t.Fatalf("unexpected messages: %+v", messages)
	}
}
