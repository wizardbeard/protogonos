package substrate

import (
	"errors"
	"math"
	"testing"
)

type invalidCEPMessage struct{}

func (invalidCEPMessage) isCEPMessage() {}

func TestBuildCEPCommandSetWeight(t *testing.T) {
	command, err := BuildCEPCommand(SetWeightCEPName, []float64{1}, map[string]float64{"scale": 0.5})
	if err != nil {
		t.Fatalf("build set_weight command: %v", err)
	}
	if command.Command != SetWeightCEPName {
		t.Fatalf("unexpected command: %s", command.Command)
	}
	if len(command.Signal) != 1 || math.Abs(command.Signal[0]-0.5) > 1e-9 {
		t.Fatalf("unexpected set_weight signal: %+v", command.Signal)
	}
	if command.FromPID != "" {
		t.Fatalf("expected BuildCEPCommand to leave sender unset, got=%q", command.FromPID)
	}
}

func TestBuildCEPCommandDeltaWeightSetIterativeAlias(t *testing.T) {
	command, err := BuildCEPCommand(DefaultCEPName, []float64{-1}, nil)
	if err != nil {
		t.Fatalf("build delta_weight command: %v", err)
	}
	if command.Command != SetIterativeCEPName {
		t.Fatalf("expected set_iterative command, got=%s", command.Command)
	}
	if len(command.Signal) != 1 || math.Abs(command.Signal[0]+1) > 1e-9 {
		t.Fatalf("unexpected delta signal: %+v", command.Signal)
	}
}

func TestBuildCEPCommandSetABCNPassThrough(t *testing.T) {
	output := []float64{0.25, 0.1, 0.2, -0.3, 0.9}
	command, err := BuildCEPCommand(SetABCNCEPName, output, nil)
	if err != nil {
		t.Fatalf("build set_abcn command: %v", err)
	}
	if command.Command != SetABCNCEPName {
		t.Fatalf("unexpected command: %s", command.Command)
	}
	if len(command.Signal) != len(output) {
		t.Fatalf("unexpected set_abcn signal width: got=%d want=%d", len(command.Signal), len(output))
	}
	for i := range output {
		if command.Signal[i] != output[i] {
			t.Fatalf("unexpected set_abcn signal[%d]: got=%v want=%v", i, command.Signal[i], output[i])
		}
	}
}

func TestApplyCEPCommandSetABCNUsesSignalCoefficients(t *testing.T) {
	next, err := ApplyCEPCommand(
		0.0,
		CEPCommand{
			Command: SetABCNCEPName,
			Signal:  []float64{1, 0.2, 0.5, -0.1, 0.8},
		},
		nil,
	)
	if err != nil {
		t.Fatalf("apply set_abcn command: %v", err)
	}
	if math.Abs(next-0.4) > 1e-9 {
		t.Fatalf("unexpected set_abcn command result: got=%v want=0.4", next)
	}
}

func TestCEPProcessForwardOrderedFanIn(t *testing.T) {
	p, err := NewCEPProcessWithID("cep_test", SetABCNCEPName, nil, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}

	if _, ready, err := p.Forward("n1", []float64{0.2}); err != nil {
		t.Fatalf("forward n1: %v", err)
	} else if ready {
		t.Fatal("unexpected ready after first fan-in message")
	}

	command, ready, err := p.Forward("n2", []float64{0.8})
	if err != nil {
		t.Fatalf("forward n2: %v", err)
	}
	if !ready {
		t.Fatal("expected ready command after full fan-in cycle")
	}
	if command.Command != SetABCNCEPName {
		t.Fatalf("unexpected command: %s", command.Command)
	}
	if command.FromPID != "cep_test" {
		t.Fatalf("unexpected sender pid: got=%q want=%q", command.FromPID, "cep_test")
	}
	// Output should preserve fan-in order [0.2, 0.8].
	if len(command.Signal) != 2 || command.Signal[0] != 0.2 || command.Signal[1] != 0.8 {
		t.Fatalf("unexpected command signal: %+v", command.Signal)
	}
}

func TestCEPProcessAssignsDefaultID(t *testing.T) {
	p, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	if p.ID() == "" {
		t.Fatal("expected default cep process id")
	}
	command, ready, err := p.Forward("n1", []float64{1})
	if err != nil {
		t.Fatalf("forward n1: %v", err)
	}
	if !ready {
		t.Fatal("expected command after single fan-in cycle")
	}
	if command.FromPID != p.ID() {
		t.Fatalf("unexpected sender pid in command: got=%q want=%q", command.FromPID, p.ID())
	}
}

func TestCEPProcessBuffersOutOfOrderFanIn(t *testing.T) {
	p, err := NewCEPProcess(SetABCNCEPName, nil, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}

	if _, ready, err := p.Forward("n2", []float64{0.8}); err != nil {
		t.Fatalf("forward n2: %v", err)
	} else if ready {
		t.Fatal("unexpected ready after out-of-order n2 message")
	}

	command, ready, err := p.Forward("n1", []float64{0.2})
	if err != nil {
		t.Fatalf("forward n1: %v", err)
	}
	if !ready {
		t.Fatal("expected ready command after matching expected sender catches up")
	}
	if len(command.Signal) != 2 || command.Signal[0] != 0.2 || command.Signal[1] != 0.8 {
		t.Fatalf("unexpected set_abcn signal from buffered out-of-order fan-in: %+v", command.Signal)
	}
}

func TestCEPProcessRejectsUnknownSender(t *testing.T) {
	p, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}

	if _, _, err := p.Forward("n3", []float64{1}); !errors.Is(err, ErrUnexpectedCEPForwardPID) {
		t.Fatalf("expected ErrUnexpectedCEPForwardPID for unknown sender, got %v", err)
	}
}

func TestCEPProcessTerminate(t *testing.T) {
	p, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	p.Terminate()
	if _, _, err := p.Forward("n1", []float64{1}); !errors.Is(err, ErrCEPProcessTerminated) {
		t.Fatalf("expected ErrCEPProcessTerminated, got %v", err)
	}
}

func TestCEPProcessHandleTerminateMessage(t *testing.T) {
	p, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	if _, _, err := p.HandleMessage(CEPTerminateMessage{FromPID: "exoself"}); err != nil {
		t.Fatalf("handle terminate message: %v", err)
	}
	if _, _, err := p.HandleMessage(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); !errors.Is(err, ErrCEPProcessTerminated) {
		t.Fatalf("expected ErrCEPProcessTerminated after terminate message, got %v", err)
	}
}

func TestCEPProcessTerminateSenderConstraint(t *testing.T) {
	p, err := NewCEPProcessWithOwner("cep_owner", "exo_owner", DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}

	if _, _, err := p.HandleMessage(CEPTerminateMessage{FromPID: "wrong"}); !errors.Is(err, ErrUnexpectedCEPTerminatePID) {
		t.Fatalf("expected ErrUnexpectedCEPTerminatePID, got %v", err)
	}
	if _, _, err := p.Forward("n1", []float64{1}); err != nil {
		t.Fatalf("expected process to remain active after wrong terminate sender, got %v", err)
	}
	if _, _, err := p.HandleMessage(CEPTerminateMessage{FromPID: "exo_owner"}); err != nil {
		t.Fatalf("terminate with expected sender: %v", err)
	}
	if _, _, err := p.Forward("n1", []float64{1}); !errors.Is(err, ErrCEPProcessTerminated) {
		t.Fatalf("expected ErrCEPProcessTerminated after owner terminate, got %v", err)
	}
}

func TestCEPProcessRejectsInvalidMessageType(t *testing.T) {
	p, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	if _, _, err := p.HandleMessage(invalidCEPMessage{}); !errors.Is(err, ErrInvalidCEPMessage) {
		t.Fatalf("expected ErrInvalidCEPMessage, got %v", err)
	}
}

func TestCEPActorForwardRoundTrip(t *testing.T) {
	process, err := NewCEPProcessWithID("cep_actor_test", SetABCNCEPName, nil, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.Terminate()
	})

	if _, ready, err := actor.Call(CEPForwardMessage{FromPID: "n2", Input: []float64{0.8}}); err != nil {
		t.Fatalf("forward n2: %v", err)
	} else if ready {
		t.Fatal("unexpected ready after first out-of-order sender")
	}

	command, ready, err := actor.Call(CEPForwardMessage{FromPID: "n1", Input: []float64{0.2}})
	if err != nil {
		t.Fatalf("forward n1: %v", err)
	}
	if !ready {
		t.Fatal("expected ready command after completing fan-in cycle")
	}
	if command.FromPID != "cep_actor_test" || command.Command != SetABCNCEPName {
		t.Fatalf("unexpected command envelope: %+v", command)
	}
	if len(command.Signal) != 2 || command.Signal[0] != 0.2 || command.Signal[1] != 0.8 {
		t.Fatalf("unexpected command signal ordering: %+v", command.Signal)
	}
}

func TestCEPActorCommandOutbox(t *testing.T) {
	process, err := NewCEPProcessWithID("cep_actor_outbox", SetABCNCEPName, nil, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.Terminate()
	})

	if _, ready, err := actor.Call(CEPForwardMessage{FromPID: "n1", Input: []float64{0.2}}); err != nil {
		t.Fatalf("forward n1: %v", err)
	} else if ready {
		t.Fatal("unexpected ready after first sender")
	}
	if _, ready, err := actor.Call(CEPForwardMessage{FromPID: "n2", Input: []float64{0.8}}); err != nil {
		t.Fatalf("forward n2: %v", err)
	} else if !ready {
		t.Fatal("expected ready after second sender")
	}

	command, err := actor.NextCommand()
	if err != nil {
		t.Fatalf("next command: %v", err)
	}
	if command.FromPID != "cep_actor_outbox" || command.Command != SetABCNCEPName {
		t.Fatalf("unexpected outbox command envelope: %+v", command)
	}
	if len(command.Signal) != 2 || command.Signal[0] != 0.2 || command.Signal[1] != 0.8 {
		t.Fatalf("unexpected outbox command signal: %+v", command.Signal)
	}
	if _, err := actor.NextCommand(); !errors.Is(err, ErrCEPActorNoCommandReady) {
		t.Fatalf("expected ErrCEPActorNoCommandReady when outbox empty, got %v", err)
	}
}

func TestCEPActorPostAndErrorMailbox(t *testing.T) {
	process, err := NewCEPProcessWithOwner("cep_actor_post", "exo_owner", DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom("exo_owner")
	})

	if err := actor.Post(CEPForwardMessage{FromPID: "n2", Input: []float64{1}}); err != nil {
		t.Fatalf("post invalid sender: %v", err)
	}
	gotErr := actor.NextError()
	if !errors.Is(gotErr, ErrUnexpectedCEPForwardPID) {
		t.Fatalf("expected ErrUnexpectedCEPForwardPID from errbox, got %v", gotErr)
	}
	if gotErr := actor.NextError(); !errors.Is(gotErr, ErrCEPActorNoError) {
		t.Fatalf("expected ErrCEPActorNoError when errbox empty, got %v", gotErr)
	}

	if err := actor.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("post valid sender: %v", err)
	}
	command, err := actor.NextCommand()
	if err != nil {
		t.Fatalf("next command after valid post: %v", err)
	}
	if command.Command != SetIterativeCEPName {
		t.Fatalf("unexpected command from post flow: %+v", command)
	}
}

func TestCEPActorInitHandshake(t *testing.T) {
	actor := NewCEPActorWithOwner("exo_owner")

	if err := actor.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("post before init: %v", err)
	}
	if gotErr := actor.NextError(); !errors.Is(gotErr, ErrCEPActorUninitialized) {
		t.Fatalf("expected ErrCEPActorUninitialized in errbox before init, got %v", gotErr)
	}

	process, err := NewCEPProcessWithOwner("cep_init", "exo_owner", DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	if _, _, err := actor.Call(CEPInitMessage{FromPID: "wrong", Process: process}); !errors.Is(err, ErrUnexpectedCEPInitPID) {
		t.Fatalf("expected ErrUnexpectedCEPInitPID, got %v", err)
	}
	if _, _, err := actor.Call(CEPInitMessage{FromPID: "exo_owner", Process: nil}); !errors.Is(err, ErrCEPActorInitProcessRequired) {
		t.Fatalf("expected ErrCEPActorInitProcessRequired, got %v", err)
	}
	if _, _, err := actor.Call(CEPInitMessage{FromPID: "exo_owner", Process: process}); err != nil {
		t.Fatalf("expected init success from owner, got %v", err)
	}
	if _, _, err := actor.Call(CEPInitMessage{FromPID: "exo_owner", Process: process}); !errors.Is(err, ErrCEPActorAlreadyInitialized) {
		t.Fatalf("expected ErrCEPActorAlreadyInitialized, got %v", err)
	}

	if err := actor.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("post after init: %v", err)
	}
	if _, err := actor.NextCommand(); err != nil {
		t.Fatalf("next command after init+post: %v", err)
	}
	if err := actor.TerminateFrom("exo_owner"); err != nil {
		t.Fatalf("terminate actor: %v", err)
	}
}

func TestCEPActorTerminateAndSubsequentCall(t *testing.T) {
	process, err := NewCEPProcess(DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)

	if err := actor.Terminate(); err != nil {
		t.Fatalf("terminate actor: %v", err)
	}
	if _, _, err := actor.Call(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); !errors.Is(err, ErrCEPActorTerminated) {
		t.Fatalf("expected ErrCEPActorTerminated after terminate, got %v", err)
	}
}

func TestCEPActorTerminateSenderConstraint(t *testing.T) {
	process, err := NewCEPProcessWithOwner("cep_actor_owner", "exo_owner", DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)

	if err := actor.TerminateFrom("wrong"); !errors.Is(err, ErrUnexpectedCEPTerminatePID) {
		t.Fatalf("expected ErrUnexpectedCEPTerminatePID, got %v", err)
	}
	if _, _, err := actor.Call(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("expected actor active after wrong terminate sender, got %v", err)
	}
	if err := actor.TerminateFrom("exo_owner"); err != nil {
		t.Fatalf("terminate with expected sender: %v", err)
	}
	if _, _, err := actor.Call(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); !errors.Is(err, ErrCEPActorTerminated) {
		t.Fatalf("expected ErrCEPActorTerminated after owner terminate, got %v", err)
	}
}
