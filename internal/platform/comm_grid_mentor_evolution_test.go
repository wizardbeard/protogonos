package platform

import (
	"context"
	"strings"
	"testing"

	"protogonos/internal/evo"
	"protogonos/internal/llm"
	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
)

type platformNoopMutation struct{}

type commGridMentorEvolutionProvider struct {
	silent bool
}

type commGridMentorEvolutionScape struct {
	plan string
}

func (platformNoopMutation) Name() string { return "noop" }

func (platformNoopMutation) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	return genome, nil
}

func (p commGridMentorEvolutionProvider) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	if p.silent || len(req.Messages) == 0 {
		return commGridMentorEvolutionHint("", 0), nil
	}
	prompt := req.Messages[0].Content
	hint := "east"
	switch {
	case strings.Contains(prompt, "position=(2,0)") && strings.Contains(prompt, "carrying_key=true"):
		hint = "drop"
	case strings.Contains(prompt, "position=(1,0)") && strings.Contains(prompt, "carrying_key=false"):
		hint = "pick"
	}
	return commGridMentorEvolutionHint(hint, 1), nil
}

func TestPolisRunEvolutionComparesCommGridMentorWithNoHintBaseline(t *testing.T) {
	initial := []model.Genome{
		commGridMentorHintGenome("mentor-g0"),
		commGridMentorHintGenome("mentor-g1"),
		commGridMentorHintGenome("mentor-g2"),
		commGridMentorHintGenome("mentor-g3"),
	}
	run := func(sc scape.Scape, runID string) EvolutionResult {
		store := storage.NewMemoryStore()
		p := NewPolis(Config{Store: store})
		if err := p.Init(context.Background()); err != nil {
			t.Fatalf("init %s: %v", runID, err)
		}
		if err := p.RegisterScape(sc); err != nil {
			t.Fatalf("register scape %s: %v", runID, err)
		}
		result, err := p.RunEvolution(context.Background(), EvolutionConfig{
			RunID:           runID,
			ScapeName:       sc.Name(),
			PopulationSize:  len(initial),
			Generations:     2,
			EliteCount:      1,
			Workers:         1,
			Seed:            77,
			InputNeuronIDs:  commGridMentorInputIDs(),
			OutputNeuronIDs: []string{"out-x", "out-y", "out-tool"},
			Mutation:        platformNoopMutation{},
			Selector:        evo.TournamentSelector{PoolSize: 0, TournamentSize: 2},
			Initial:         initial,
		})
		if err != nil {
			t.Fatalf("run evolution %s: %v", runID, err)
		}
		if len(result.BestByGeneration) != 2 {
			t.Fatalf("expected two generations for %s, got %d", runID, len(result.BestByGeneration))
		}
		if len(result.Lineage) == 0 {
			t.Fatalf("expected lineage for %s", runID)
		}
		return result
	}

	mentor := run(commGridMentorEvolutionScape{plan: "solve"}, "comm-grid-mentor-evo")
	baseline := run(commGridMentorEvolutionScape{plan: "silent"}, "comm-grid-mentor-baseline-evo")

	if mentor.BestFinalFitness <= baseline.BestFinalFitness {
		t.Fatalf("expected mentor to beat no-hint baseline: mentor=%f baseline=%f", mentor.BestFinalFitness, baseline.BestFinalFitness)
	}
	if mentor.BestFinalFitness < 1.0 {
		t.Fatalf("expected mentor run to solve task, fitness=%f history=%+v", mentor.BestFinalFitness, mentor.BestByGeneration)
	}
	if baseline.BestFinalFitness != 0 {
		t.Fatalf("expected no-hint baseline to stay at zero, fitness=%f history=%+v", baseline.BestFinalFitness, baseline.BestByGeneration)
	}
}

func (s commGridMentorEvolutionScape) Name() string {
	return "comm-grid-mentor-evo"
}

func (s commGridMentorEvolutionScape) Evaluate(ctx context.Context, agent scape.Agent) (scape.Fitness, scape.Trace, error) {
	mentor := scape.CommGridMentorScape{Config: scape.CommGridMentorConfig{
		CommGridConfig: scape.CommGridConfig{
			Width:    3,
			Height:   3,
			MaxSteps: 8,
			Key:      scape.CommGridPoint{X: 1, Y: 0},
			Goal:     scape.CommGridPoint{X: 2, Y: 0},
			Agents: []scape.CommGridAgentState{{
				ID:       agent.ID(),
				Position: scape.CommGridPoint{},
			}},
		},
		Provider: commGridMentorEvolutionProvider{silent: s.plan == "silent"},
		Model:    "fixture-mentor",
	}}
	return mentor.Evaluate(ctx, agent)
}

func commGridMentorEvolutionHint(hint string, tokens int) llm.Response {
	return llm.Response{
		Message:      hint,
		Usage:        llm.Usage{TotalTokens: tokens},
		FinishReason: "stop",
		Model:        "fixture-mentor",
	}
}

func commGridMentorHintGenome(id string) model.Genome {
	inputs := commGridMentorInputIDs()
	neurons := make([]model.Neuron, 0, len(inputs)+3)
	for _, inputID := range inputs {
		neurons = append(neurons, model.Neuron{ID: inputID, Activation: "identity"})
	}
	neurons = append(neurons,
		model.Neuron{ID: "out-x", Activation: "identity"},
		model.Neuron{ID: "out-y", Activation: "identity"},
		model.Neuron{ID: "out-tool", Activation: "identity"},
	)
	return model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
		ID:              id,
		Neurons:         neurons,
		Synapses: []model.Synapse{
			{ID: id + "-hint-x", From: "hint-x", To: "out-x", Weight: 1, Enabled: true},
			{ID: id + "-hint-y", From: "hint-y", To: "out-y", Weight: 1, Enabled: true},
			{ID: id + "-hint-tool", From: "hint-tool", To: "out-tool", Weight: 1, Enabled: true},
		},
	}
}

func commGridMentorInputIDs() []string {
	return []string{"target-x", "target-y", "carrying", "step", "hint-x", "hint-y", "hint-tool", "hint-valid"}
}
