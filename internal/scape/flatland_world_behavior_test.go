package scape

import "testing"

func TestFlatlandWorldBehaviorPreyPlantPoisonBranches(t *testing.T) {
	prey := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 100}
	plant := FlatlandAvatar{Type: FlatlandObjectPlant, Energy: 500}
	poison := FlatlandAvatar{Type: FlatlandObjectPoison, Energy: -2000}

	eaten := FlatlandWorldBehavior(true, false, prey, plant)
	if eaten.Order != FlatlandWorldOrderPlantEaten || eaten.EnergyDelta != 500 {
		t.Fatalf("expected full plant energy, got %+v", eaten)
	}

	prey.Spear = true
	spearEaten := FlatlandWorldBehavior(true, false, prey, plant)
	if spearEaten.Order != FlatlandWorldOrderPlantEaten || spearEaten.EnergyDelta != 100 {
		t.Fatalf("expected spear-scaled plant energy, got %+v", spearEaten)
	}

	poisoned := FlatlandWorldBehavior(true, false, prey, poison)
	if poisoned.Order != FlatlandWorldOrderPoisonEaten || poisoned.EnergyDelta != -2000 {
		t.Fatalf("expected poison energy penalty, got %+v", poisoned)
	}
}

func TestFlatlandWorldBehaviorPreyPenetrationAndCollisionBranches(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 1000, Location: FlatlandPoint{}, Radius: 3}
	target := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 100, Location: FlatlandPoint{X: 4}, Radius: 3}

	destroyPrey := FlatlandWorldBehavior(false, true, operator, target)
	if destroyPrey.Order != FlatlandWorldOrderDestroy || destroyPrey.EnergyDelta != 500 {
		t.Fatalf("expected prey destroy energy, got %+v", destroyPrey)
	}

	target.Spear = true
	destroyPredatorLikePrey := FlatlandWorldBehavior(false, true, operator, target)
	if destroyPredatorLikePrey.Order != FlatlandWorldOrderDestroy || destroyPredatorLikePrey.EnergyDelta != 100 {
		t.Fatalf("expected predator-like prey destroy energy, got %+v", destroyPredatorLikePrey)
	}

	target.Spear = false
	pushed := FlatlandWorldBehavior(true, false, operator, target)
	if pushed.Order != FlatlandWorldOrderVoid || pushed.Target.Location.X <= target.Location.X {
		t.Fatalf("expected prey collision push, got %+v", pushed)
	}
	assertClose(t, "prey collision push energy", pushed.Target.Energy, 99)
}

func TestFlatlandWorldBehaviorPredatorBranches(t *testing.T) {
	predator := FlatlandAvatar{Type: FlatlandObjectPredator, Energy: 1000, Location: FlatlandPoint{}, Radius: 3}
	prey := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 100, Location: FlatlandPoint{X: 4}, Radius: 3}

	destroy := FlatlandWorldBehavior(false, true, predator, prey)
	if destroy.Order != FlatlandWorldOrderDestroy || destroy.EnergyDelta != 500 {
		t.Fatalf("expected predator penetration destroy, got %+v", destroy)
	}

	pushedPrey := FlatlandWorldBehavior(true, false, predator, prey)
	if pushedPrey.Order != FlatlandWorldOrderVoid || pushedPrey.Target.Location.X <= prey.Location.X {
		t.Fatalf("expected predator-prey collision push, got %+v", pushedPrey)
	}
	assertClose(t, "predator-prey push cost", pushedPrey.Target.Energy, 90)

	otherPredator := FlatlandAvatar{Type: FlatlandObjectPredator, Energy: 100, Location: FlatlandPoint{X: 4}, Radius: 3}
	pushedPredator := FlatlandWorldBehavior(false, true, predator, otherPredator)
	if pushedPredator.Order != FlatlandWorldOrderVoid || pushedPredator.Target.Location.X <= otherPredator.Location.X {
		t.Fatalf("expected predator penetration push, got %+v", pushedPredator)
	}
	assertClose(t, "predator penetration push cost", pushedPredator.Target.Energy, 90)
}

func TestFlatlandWorldBehaviorObstacleBranches(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 1000, Location: FlatlandPoint{}, Radius: 3}
	rock := FlatlandAvatar{Type: FlatlandObjectRock, Energy: 10, Location: FlatlandPoint{X: 4}, Radius: 3}
	firePit := FlatlandAvatar{Type: FlatlandObjectFirePit, Energy: 10, Location: FlatlandPoint{X: 4}, Radius: 3}
	beacon := FlatlandAvatar{Type: FlatlandObjectBeacon, Energy: 10, Location: FlatlandPoint{X: 4}, Radius: 3}

	rocked := FlatlandWorldBehavior(true, false, operator, rock)
	if rocked.Order != FlatlandWorldOrderVoid || rocked.EnergyDelta != -1 || rocked.Target.Location.X <= rock.Location.X {
		t.Fatalf("expected rock penalty and push, got %+v", rocked)
	}

	burned := FlatlandWorldBehavior(true, false, operator, firePit)
	if burned.EnergyDelta != -100 || burned.Target.Location.X <= firePit.Location.X {
		t.Fatalf("expected fire-pit penalty and push, got %+v", burned)
	}

	signaled := FlatlandWorldBehavior(true, false, operator, beacon)
	if signaled.EnergyDelta != 0 || signaled.Target.Location.X <= beacon.Location.X {
		t.Fatalf("expected beacon push without energy delta, got %+v", signaled)
	}
}

func TestFlatlandWorldBehaviorStrongerObstaclePushesOperator(t *testing.T) {
	operator := FlatlandAvatar{Type: FlatlandObjectPrey, Energy: 10, Location: FlatlandPoint{X: 4}, Radius: 3}
	rock := FlatlandAvatar{Type: FlatlandObjectRock, Energy: 1000, Location: FlatlandPoint{}, Radius: 3}

	result := FlatlandWorldBehavior(true, false, operator, rock)
	if result.EnergyDelta != -1 || result.Operator.Location.X <= operator.Location.X {
		t.Fatalf("expected stronger obstacle to push operator, got %+v", result)
	}
	assertClose(t, "zero-strength push no energy cost", result.Operator.Energy, operator.Energy)
}
