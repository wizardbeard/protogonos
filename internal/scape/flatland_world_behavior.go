package scape

const (
	FlatlandWorldOrderVoid        = "void"
	FlatlandWorldOrderDestroy     = "destroy"
	FlatlandWorldOrderPlantEaten  = "plant_eaten"
	FlatlandWorldOrderPoisonEaten = "poison_eaten"
)

type FlatlandWorldBehaviorResult struct {
	EnergyDelta float64
	Order       string
	Operator    FlatlandAvatar
	Target      FlatlandAvatar
}

func FlatlandWorldBehavior(collision, penetration bool, operator, target FlatlandAvatar) FlatlandWorldBehaviorResult {
	switch {
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPlant && operator.Spear:
		return flatlandWorldBehaviorResult(target.Energy*0.2, FlatlandWorldOrderPlantEaten, operator, target)
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPlant:
		return flatlandWorldBehaviorResult(target.Energy, FlatlandWorldOrderPlantEaten, operator, target)
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPoison:
		return flatlandWorldBehaviorResult(target.Energy, FlatlandWorldOrderPoisonEaten, operator, target)
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPrey && penetration && !target.Spear:
		return flatlandWorldBehaviorResult(500, FlatlandWorldOrderDestroy, operator, target)
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPrey && penetration && target.Spear:
		return flatlandWorldBehaviorResult(100, FlatlandWorldOrderDestroy, operator, target)
	case operator.Type == FlatlandObjectPrey && target.Type == FlatlandObjectPrey && collision:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 0.1))
	case operator.Type == FlatlandObjectPredator && target.Type == FlatlandObjectPrey && penetration:
		return flatlandWorldBehaviorResult(500, FlatlandWorldOrderDestroy, operator, target)
	case operator.Type == FlatlandObjectPredator && target.Type == FlatlandObjectPrey && collision:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 1))
	case operator.Type == FlatlandObjectPredator && target.Type == FlatlandObjectPredator && penetration:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 1))
	case operator.Type == FlatlandObjectPredator && target.Type == FlatlandObjectPredator && collision:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 0.1))
	case operator.Type == FlatlandObjectPredator && (target.Type == FlatlandObjectPlant || target.Type == FlatlandObjectPoison) && collision:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 0))
	case target.Type == FlatlandObjectRock:
		return flatlandPushedObstacleBehavior(-1, operator, target)
	case target.Type == FlatlandObjectFirePit:
		return flatlandPushedObstacleBehavior(-100, operator, target)
	case target.Type == FlatlandObjectBeacon:
		return flatlandPushedObstacleBehavior(0, operator, target)
	default:
		return flatlandWorldBehaviorResult(0, FlatlandWorldOrderVoid, operator, target)
	}
}

func flatlandPushedObstacleBehavior(energyDelta float64, operator, target FlatlandAvatar) FlatlandWorldBehaviorResult {
	if operator.Energy > target.Energy {
		return flatlandWorldBehaviorResult(energyDelta, FlatlandWorldOrderVoid, operator, FlatlandPush(operator, target, 1))
	}
	return flatlandWorldBehaviorResult(energyDelta, FlatlandWorldOrderVoid, FlatlandPush(target, operator, 0), target)
}

func flatlandWorldBehaviorResult(energyDelta float64, order string, operator, target FlatlandAvatar) FlatlandWorldBehaviorResult {
	return FlatlandWorldBehaviorResult{
		EnergyDelta: energyDelta,
		Order:       order,
		Operator:    operator,
		Target:      target,
	}
}
