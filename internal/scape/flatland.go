package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

type FlatlandScape struct{}

func (FlatlandScape) Name() string {
	return "flatland"
}

func (FlatlandScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return FlatlandScape{}.EvaluateMode(ctx, agent, "gt")
}

func (FlatlandScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := flatlandConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateFlatlandWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateFlatlandWithStep(ctx, runner, cfg)
}

type flatlandModeConfig struct {
	mode            string
	maxAge          int
	forageGoal      int
	foodPositions   []int
	poisonPositions []int
	wallPositions   []int
}

func flatlandConfigForMode(mode string) (flatlandModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return flatlandModeConfig{
			mode:            "gt",
			maxAge:          flatlandDefaultMaxAge,
			forageGoal:      flatlandDefaultForageGoal,
			foodPositions:   []int{5, 11, 19, 27, 35, 43},
			poisonPositions: []int{14, 30},
			wallPositions:   []int{8, 16, 24, 32, 40},
		}, nil
	case "validation":
		return flatlandModeConfig{
			mode:            "validation",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{4, 10, 18, 26, 34, 42},
			poisonPositions: []int{13, 29},
			wallPositions:   []int{7, 15, 23, 31, 39},
		}, nil
	case "test":
		return flatlandModeConfig{
			mode:            "test",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{6, 12, 20, 28, 36, 44},
			poisonPositions: []int{15, 31},
			wallPositions:   []int{9, 17, 25, 33, 41},
		}, nil
	case "benchmark":
		return flatlandModeConfig{
			mode:            "benchmark",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{6, 12, 20, 28, 36, 44},
			poisonPositions: []int{15, 31},
			wallPositions:   []int{9, 17, 25, 33, 41},
		}, nil
	default:
		return flatlandModeConfig{}, fmt.Errorf("unsupported flatland mode: %s", mode)
	}
}

func evaluateFlatlandWithStep(ctx context.Context, runner StepAgent, cfg flatlandModeConfig) (Fitness, Trace, error) {
	fitness, trace, err := evaluateFlatland(ctx, cfg, func(ctx context.Context, sense flatlandSenseInput) (flatlandControl, error) {
		// Preserve manual StepAgent compatibility with the original 2-channel flatland input.
		out, err := runner.RunStep(ctx, []float64{sense.distance, sense.energy})
		if err != nil {
			return flatlandControl{}, err
		}
		return flatlandControlFromOutput(out)
	})
	if err != nil {
		return 0, nil, err
	}
	trace["control_surface"] = "step_output"
	return fitness, trace, nil
}

func evaluateFlatlandWithTick(ctx context.Context, ticker TickAgent, cfg flatlandModeConfig) (Fitness, Trace, error) {
	ioBindings, err := flatlandIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	fitness, trace, err := evaluateFlatland(ctx, cfg, func(ctx context.Context, sense flatlandSenseInput) (flatlandControl, error) {
		if ioBindings.distanceSetter != nil {
			ioBindings.distanceSetter.Set(sense.distance)
		}
		if ioBindings.energySetter != nil {
			ioBindings.energySetter.Set(sense.energy)
		}
		if ioBindings.poisonSetter != nil {
			ioBindings.poisonSetter.Set(sense.poison)
		}
		if ioBindings.wallSetter != nil {
			ioBindings.wallSetter.Set(sense.wall)
		}
		if ioBindings.foodProximitySetter != nil {
			ioBindings.foodProximitySetter.Set(sense.foodProximity)
		}
		if ioBindings.poisonProximitySetter != nil {
			ioBindings.poisonProximitySetter.Set(sense.poisonProximity)
		}
		if ioBindings.wallProximitySetter != nil {
			ioBindings.wallProximitySetter.Set(sense.wallProximity)
		}
		if ioBindings.resourceBalanceSetter != nil {
			ioBindings.resourceBalanceSetter.Set(sense.resourceBalance)
		}
		for i := range ioBindings.distanceScanSetters {
			if ioBindings.distanceScanSetters[i] != nil {
				ioBindings.distanceScanSetters[i].Set(sense.distanceScan[i])
			}
		}
		for i := range ioBindings.colorScanSetters {
			if ioBindings.colorScanSetters[i] != nil {
				ioBindings.colorScanSetters[i].Set(sense.colorScan[i])
			}
		}
		for i := range ioBindings.energyScanSetters {
			if ioBindings.energyScanSetters[i] != nil {
				ioBindings.energyScanSetters[i].Set(sense.energyScan[i])
			}
		}
		out, err := ticker.Tick(ctx)
		if err != nil {
			return flatlandControl{}, err
		}
		values := ioBindings.moveOutput.Last()
		if len(values) == 0 {
			values = out
		}
		return flatlandControlFromOutput(values)
	})
	if err != nil {
		return 0, nil, err
	}
	trace["control_surface"] = ioBindings.controlSurface
	return fitness, trace, nil
}

func evaluateFlatland(
	ctx context.Context,
	cfg flatlandModeConfig,
	chooseMove func(context.Context, flatlandSenseInput) (flatlandControl, error),
) (Fitness, Trace, error) {
	episode := newFlatlandEpisode(cfg)
	movementSteps := 0
	foodCollisions := 0
	poisonCollisions := 0
	lastDistance := 0.0
	lastPoison := 0.0
	lastWall := 0.0
	lastFoodProximity := 0.0
	lastPoisonProximity := 0.0
	lastWallProximity := 0.0
	lastResourceBalance := 0.0
	lastDistanceScanMean := 0.0
	lastColorScanMean := 0.0
	lastEnergyScanMean := 0.0
	lastControlWidth := 0
	terminalReason := "age_limit"

	for episode.age < cfg.maxAge && episode.energy > 0 {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		episode.advanceRespawns()
		sense := episode.sense()
		lastDistance = sense.distance
		lastPoison = sense.poison
		lastWall = sense.wall
		lastFoodProximity = sense.foodProximity
		lastPoisonProximity = sense.poisonProximity
		lastWallProximity = sense.wallProximity
		lastResourceBalance = sense.resourceBalance
		lastDistanceScanMean = meanFlatlandScan(sense.distanceScan)
		lastColorScanMean = meanFlatlandScan(sense.colorScan)
		lastEnergyScanMean = meanFlatlandScan(sense.energyScan)

		control, err := chooseMove(ctx, sense)
		if err != nil {
			return 0, nil, err
		}
		lastControlWidth = control.width

		moveStep, hitFood, hitPoison, _, reason := episode.step(control.move)
		if moveStep != 0 {
			movementSteps++
		}
		if hitFood {
			foodCollisions++
		}
		if hitPoison {
			poisonCollisions++
		}
		if reason != "" {
			terminalReason = reason
			break
		}
	}

	if episode.energy <= 0 {
		terminalReason = "depleted"
	}
	if episode.foodCollected >= episode.forageGoal {
		terminalReason = "forage_goal"
	}

	age := episode.age
	if age <= 0 {
		age = 1
	}

	resourceCollisions := foodCollisions + poisonCollisions
	totalCollisions := resourceCollisions + episode.wallCollisions
	avgReward := episode.rewardAcc / float64(age)
	survival := float64(episode.age) / float64(cfg.maxAge)
	energyTerm := clamp(episode.normalizedEnergy(), 0, 1)
	forageDenom := float64(resourceCollisions + 2)
	forageBalance := float64(episode.foodCollected-episode.poisonHits) / forageDenom
	forageTerm := 0.5 + 0.5*clamp(forageBalance, -1, 1)
	rewardTerm := 0.5 + 0.5*clamp(avgReward, -1, 1)
	wallPenalty := clamp(float64(episode.wallCollisions)/float64(age), 0, 1)
	respawnActivity := clamp(float64(episode.resourceRespawns)/float64(age), 0, 1)

	fitness := 0.33*survival + 0.24*energyTerm + 0.24*forageTerm + 0.14*rewardTerm - 0.12*wallPenalty + 0.03*respawnActivity
	if episode.foodCollected >= episode.forageGoal {
		fitness += 0.1
	}
	fitness = clamp(fitness, 0, 1.4)

	return Fitness(fitness), Trace{
		"position":                float64(episode.position),
		"energy":                  episode.energy,
		"energy_norm":             episode.normalizedEnergy(),
		"reward":                  avgReward,
		"reward_total":            episode.rewardAcc,
		"age":                     episode.age,
		"max_age":                 cfg.maxAge,
		"forage_goal":             episode.forageGoal,
		"food_collected":          episode.foodCollected,
		"poison_hits":             episode.poisonHits,
		"collisions":              totalCollisions,
		"wall_collisions":         episode.wallCollisions,
		"movement_steps":          movementSteps,
		"resource_respawns":       episode.resourceRespawns,
		"active_food":             episode.activeResources(episode.food),
		"active_poison":           episode.activeResources(episode.poison),
		"terminal_reason":         terminalReason,
		"last_food_distance":      lastDistance,
		"last_poison_signal":      lastPoison,
		"last_wall_signal":        lastWall,
		"last_food_proximity":     lastFoodProximity,
		"last_poison_proximity":   lastPoisonProximity,
		"last_wall_proximity":     lastWallProximity,
		"last_resource_balance":   lastResourceBalance,
		"last_distance_scan_mean": lastDistanceScanMean,
		"last_color_scan_mean":    lastColorScanMean,
		"last_energy_scan_mean":   lastEnergyScanMean,
		"last_control_width":      lastControlWidth,
		"feature_width":           8,
		"scanner_density":         flatlandScannerDensity,
		"scanner_feature_width":   flatlandScannerWidth,
		"mode":                    cfg.mode,
	}, nil
}

const (
	flatlandWorldSize         = 48
	flatlandDefaultMaxAge     = 220
	flatlandInitialEnergy     = 1.2
	flatlandEnergyCap         = 2.0
	flatlandBaseMetabolic     = 0.012
	flatlandIdleMetabolic     = 0.004
	flatlandMoveMetabolic     = 0.018
	flatlandWallEnergyPenalty = 0.035
	flatlandFoodEnergyMin     = 0.20
	flatlandFoodEnergyMax     = 0.38
	flatlandFoodGrowth        = 0.012
	flatlandPoisonDamageMin   = 0.30
	flatlandPoisonDamageMax   = 0.44
	flatlandPoisonDrift       = 0.006
	flatlandSurvivalReward    = 0.01
	flatlandFoodReward        = 0.25
	flatlandPoisonPenalty     = 0.30
	flatlandWallPenalty       = 0.07
	flatlandFoodRespawn       = 12
	flatlandPoisonRespawn     = 16
	flatlandDefaultForageGoal = 8
	flatlandSensorDistanceLo  = -1.0
	flatlandSensorDistanceHi  = 1.0
	flatlandRespawnStride     = 7
	flatlandScannerDensity    = 5
	flatlandScannerWidth      = flatlandScannerDensity * 3
)

var flatlandDistanceScannerSensors = [flatlandScannerDensity]string{
	protoio.FlatlandDistanceScan0SensorName,
	protoio.FlatlandDistanceScan1SensorName,
	protoio.FlatlandDistanceScan2SensorName,
	protoio.FlatlandDistanceScan3SensorName,
	protoio.FlatlandDistanceScan4SensorName,
}

var flatlandColorScannerSensors = [flatlandScannerDensity]string{
	protoio.FlatlandColorScan0SensorName,
	protoio.FlatlandColorScan1SensorName,
	protoio.FlatlandColorScan2SensorName,
	protoio.FlatlandColorScan3SensorName,
	protoio.FlatlandColorScan4SensorName,
}

var flatlandEnergyScannerSensors = [flatlandScannerDensity]string{
	protoio.FlatlandEnergyScan0SensorName,
	protoio.FlatlandEnergyScan1SensorName,
	protoio.FlatlandEnergyScan2SensorName,
	protoio.FlatlandEnergyScan3SensorName,
	protoio.FlatlandEnergyScan4SensorName,
}

type flatlandResource struct {
	position int
	cooldown int
	potency  float64
}

type flatlandSenseInput struct {
	distance        float64
	energy          float64
	poison          float64
	wall            float64
	foodProximity   float64
	poisonProximity float64
	wallProximity   float64
	resourceBalance float64
	distanceScan    [flatlandScannerDensity]float64
	colorScan       [flatlandScannerDensity]float64
	energyScan      [flatlandScannerDensity]float64
}

type flatlandControl struct {
	move  float64
	width int
}

type flatlandEpisode struct {
	position         int
	energy           float64
	age              int
	maxAge           int
	forageGoal       int
	foodCollected    int
	poisonHits       int
	wallCollisions   int
	resourceRespawns int
	rewardAcc        float64
	food             []flatlandResource
	poison           []flatlandResource
	walls            map[int]struct{}
	respawnCursor    int
}

func newFlatlandEpisode(cfg flatlandModeConfig) *flatlandEpisode {
	maxAge := cfg.maxAge
	if maxAge <= 0 {
		maxAge = flatlandDefaultMaxAge
	}
	forageGoal := cfg.forageGoal
	if forageGoal <= 0 {
		forageGoal = flatlandDefaultForageGoal
	}

	foodPositions := cfg.foodPositions
	if len(foodPositions) == 0 {
		foodPositions = []int{5, 11, 19, 27, 35, 43}
	}
	poisonPositions := cfg.poisonPositions
	if len(poisonPositions) == 0 {
		poisonPositions = []int{14, 30}
	}
	wallPositions := cfg.wallPositions
	if len(wallPositions) == 0 {
		wallPositions = []int{8, 16, 24, 32, 40}
	}

	walls := make(map[int]struct{}, len(wallPositions))
	for _, position := range wallPositions {
		walls[wrapFlatlandPosition(position)] = struct{}{}
	}

	occupied := make(map[int]struct{})
	food := buildFlatlandResources(foodPositions, flatlandFoodEnergyMin, walls, occupied)
	poison := buildFlatlandResources(poisonPositions, flatlandPoisonDamageMin, walls, occupied)

	start := 0
	for offset := 0; offset < flatlandWorldSize; offset++ {
		candidate := wrapFlatlandPosition(offset * flatlandRespawnStride)
		if _, blocked := walls[candidate]; blocked {
			continue
		}
		if _, taken := occupied[candidate]; taken {
			continue
		}
		start = candidate
		break
	}

	return &flatlandEpisode{
		position:      start,
		energy:        flatlandInitialEnergy,
		maxAge:        maxAge,
		forageGoal:    forageGoal,
		food:          food,
		poison:        poison,
		walls:         walls,
		respawnCursor: wrapFlatlandPosition(start + 3),
	}
}

func buildFlatlandResources(
	positions []int,
	potency float64,
	walls map[int]struct{},
	occupied map[int]struct{},
) []flatlandResource {
	out := make([]flatlandResource, 0, len(positions))
	for _, raw := range positions {
		position := wrapFlatlandPosition(raw)
		if _, blocked := walls[position]; blocked {
			continue
		}
		if _, taken := occupied[position]; taken {
			continue
		}
		occupied[position] = struct{}{}
		out = append(out, flatlandResource{position: position, potency: potency})
	}
	if len(out) > 0 {
		return out
	}
	for offset := 0; offset < flatlandWorldSize && len(out) < 2; offset++ {
		candidate := wrapFlatlandPosition(offset*flatlandRespawnStride + 1)
		if _, blocked := walls[candidate]; blocked {
			continue
		}
		if _, taken := occupied[candidate]; taken {
			continue
		}
		occupied[candidate] = struct{}{}
		out = append(out, flatlandResource{position: candidate, potency: potency})
	}
	return out
}

func (e *flatlandEpisode) advanceRespawns() {
	for i := range e.food {
		resource := &e.food[i]
		if resource.cooldown > 0 {
			resource.cooldown--
			if resource.cooldown == 0 {
				resource.position = e.nextRespawnPosition(resource.position)
				resource.potency = flatlandFoodEnergyMin
				e.resourceRespawns++
			}
			continue
		}
		resource.potency = clamp(resource.potency+flatlandFoodGrowth, flatlandFoodEnergyMin, flatlandFoodEnergyMax)
	}

	for i := range e.poison {
		resource := &e.poison[i]
		if resource.cooldown > 0 {
			resource.cooldown--
			if resource.cooldown == 0 {
				resource.position = e.nextRespawnPosition(resource.position)
				resource.potency = flatlandPoisonDamageMin
				e.resourceRespawns++
			}
			continue
		}
		resource.potency = clamp(resource.potency+flatlandPoisonDrift, flatlandPoisonDamageMin, flatlandPoisonDamageMax)
	}
}

func (e *flatlandEpisode) nextRespawnPosition(previous int) int {
	occupied := make(map[int]struct{}, len(e.food)+len(e.poison)+1)
	for _, resource := range e.food {
		if resource.cooldown == 0 {
			occupied[resource.position] = struct{}{}
		}
	}
	for _, resource := range e.poison {
		if resource.cooldown == 0 {
			occupied[resource.position] = struct{}{}
		}
	}
	occupied[e.position] = struct{}{}
	delete(occupied, previous)

	for attempt := 0; attempt < flatlandWorldSize; attempt++ {
		candidate := wrapFlatlandPosition(e.respawnCursor + attempt*flatlandRespawnStride + attempt*attempt)
		if e.isWall(candidate) {
			continue
		}
		if _, taken := occupied[candidate]; taken {
			continue
		}
		e.respawnCursor = wrapFlatlandPosition(candidate + flatlandRespawnStride)
		return candidate
	}
	return previous
}

func (e *flatlandEpisode) normalizedEnergy() float64 {
	if flatlandInitialEnergy <= 0 {
		return 0
	}
	return clamp(e.energy/flatlandInitialEnergy, 0, flatlandEnergyCap/flatlandInitialEnergy)
}

func (e *flatlandEpisode) sense() flatlandSenseInput {
	distanceScan, colorScan, energyScan := e.senseScannerVectors()
	return flatlandSenseInput{
		distance:        e.senseDistanceToFood(),
		energy:          e.normalizedEnergy(),
		poison:          e.senseResourceHeading(e.poison),
		wall:            e.senseWallHeading(),
		foodProximity:   e.senseResourceProximity(e.food),
		poisonProximity: e.senseResourceProximity(e.poison),
		wallProximity:   e.senseWallProximity(),
		resourceBalance: e.senseResourceBalance(),
		distanceScan:    distanceScan,
		colorScan:       colorScan,
		energyScan:      energyScan,
	}
}

type flatlandScannerEntityKind int

const (
	flatlandScannerEntityNone flatlandScannerEntityKind = iota
	flatlandScannerEntityFood
	flatlandScannerEntityPoison
	flatlandScannerEntityWall
)

func (e *flatlandEpisode) senseScannerVectors() (
	[flatlandScannerDensity]float64,
	[flatlandScannerDensity]float64,
	[flatlandScannerDensity]float64,
) {
	var distanceScan [flatlandScannerDensity]float64
	var colorScan [flatlandScannerDensity]float64
	var energyScan [flatlandScannerDensity]float64

	half := flatlandWorldSize / 2
	if half <= 0 {
		return distanceScan, colorScan, energyScan
	}

	for i := 0; i < flatlandScannerDensity; i++ {
		probe := wrapFlatlandPosition(e.position + i - flatlandScannerDensity/2)
		kind, distance, potency, ok := e.nearestEntityFrom(probe)
		if !ok {
			continue
		}
		distanceScan[i] = clamp(1.0-float64(distance)/float64(half), 0, 1)
		switch kind {
		case flatlandScannerEntityFood:
			colorScan[i] = 1.0
			energyScan[i] = clamp(potency/flatlandFoodEnergyMax, 0, 1)
		case flatlandScannerEntityPoison:
			colorScan[i] = -1.0
			energyScan[i] = -clamp(potency/flatlandPoisonDamageMax, 0, 1)
		case flatlandScannerEntityWall:
			colorScan[i] = 0.35
			energyScan[i] = -flatlandWallPenalty
		}
	}

	return distanceScan, colorScan, energyScan
}

func (e *flatlandEpisode) nearestEntityFrom(origin int) (flatlandScannerEntityKind, int, float64, bool) {
	bestDistance := flatlandWorldSize + 1
	bestKind := flatlandScannerEntityNone
	bestPotency := 0.0

	for _, resource := range e.food {
		if resource.cooldown > 0 {
			continue
		}
		distance := absInt(signedRingDistance(origin, resource.position, flatlandWorldSize))
		if distance < bestDistance {
			bestDistance = distance
			bestKind = flatlandScannerEntityFood
			bestPotency = clamp(resource.potency, flatlandFoodEnergyMin, flatlandFoodEnergyMax)
		}
	}
	for _, resource := range e.poison {
		if resource.cooldown > 0 {
			continue
		}
		distance := absInt(signedRingDistance(origin, resource.position, flatlandWorldSize))
		if distance < bestDistance {
			bestDistance = distance
			bestKind = flatlandScannerEntityPoison
			bestPotency = clamp(resource.potency, flatlandPoisonDamageMin, flatlandPoisonDamageMax)
		}
	}
	for wall := range e.walls {
		distance := absInt(signedRingDistance(origin, wall, flatlandWorldSize))
		if distance < bestDistance {
			bestDistance = distance
			bestKind = flatlandScannerEntityWall
			bestPotency = flatlandWallPenalty
		}
	}

	if bestDistance > flatlandWorldSize || bestKind == flatlandScannerEntityNone {
		return flatlandScannerEntityNone, 0, 0, false
	}
	return bestKind, bestDistance, bestPotency, true
}

func (e *flatlandEpisode) senseDistanceToFood() float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}

	foodDelta, foodDistance, ok := e.resourceSignalDelta(e.food)
	if !ok {
		return 0
	}
	foodSignal := float64(foodDelta) / float64(half)
	signal := foodSignal

	if poisonDelta, poisonDistance, ok := e.resourceSignalDelta(e.poison); ok && poisonDistance <= foodDistance+2 {
		poisonSignal := float64(poisonDelta) / float64(half)
		signal -= 0.6 * poisonSignal
	}

	if wallDelta, wallDistance, ok := e.nearestWallDelta(); ok && wallDistance <= 2 {
		wallSignal := float64(wallDelta) / float64(half)
		signal -= 0.3 * wallSignal
	}

	return clamp(signal, flatlandSensorDistanceLo, flatlandSensorDistanceHi)
}

func (e *flatlandEpisode) senseResourceHeading(resources []flatlandResource) float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}
	delta, _, ok := e.resourceSignalDelta(resources)
	if !ok {
		return 0
	}
	return clamp(float64(delta)/float64(half), flatlandSensorDistanceLo, flatlandSensorDistanceHi)
}

func (e *flatlandEpisode) senseWallHeading() float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}
	delta, _, ok := e.nearestWallDelta()
	if !ok {
		return 0
	}
	return clamp(float64(delta)/float64(half), flatlandSensorDistanceLo, flatlandSensorDistanceHi)
}

func (e *flatlandEpisode) senseResourceProximity(resources []flatlandResource) float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}
	_, distance, ok := e.resourceSignalDelta(resources)
	if !ok {
		return 0
	}
	return clamp(1.0-float64(distance)/float64(half), 0, 1)
}

func (e *flatlandEpisode) senseWallProximity() float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}
	_, distance, ok := e.nearestWallDelta()
	if !ok {
		return 0
	}
	return clamp(1.0-float64(distance)/float64(half), 0, 1)
}

func (e *flatlandEpisode) senseResourceBalance() float64 {
	activeFood := e.activeResources(e.food)
	activePoison := e.activeResources(e.poison)
	denom := activeFood + activePoison
	if denom == 0 {
		return 0
	}
	return clamp(float64(activeFood-activePoison)/float64(denom), -1, 1)
}

func (e *flatlandEpisode) resourceSignalDelta(resources []flatlandResource) (int, int, bool) {
	bestDelta := 0
	bestDistance := flatlandWorldSize + 1
	for _, resource := range resources {
		if resource.cooldown > 0 {
			continue
		}
		delta := signedRingDistance(e.position, resource.position, flatlandWorldSize)
		distance := absInt(delta)
		if distance < bestDistance {
			bestDistance = distance
			bestDelta = delta
		}
	}
	if bestDistance > flatlandWorldSize {
		return 0, 0, false
	}
	return bestDelta, bestDistance, true
}

func (e *flatlandEpisode) nearestWallDelta() (int, int, bool) {
	if len(e.walls) == 0 {
		return 0, 0, false
	}
	bestDelta := 0
	bestDistance := flatlandWorldSize + 1
	for wall := range e.walls {
		delta := signedRingDistance(e.position, wall, flatlandWorldSize)
		distance := absInt(delta)
		if distance < bestDistance {
			bestDistance = distance
			bestDelta = delta
		}
	}
	if bestDistance > flatlandWorldSize {
		return 0, 0, false
	}
	return bestDelta, bestDistance, true
}

func (e *flatlandEpisode) activeResources(resources []flatlandResource) int {
	count := 0
	for _, resource := range resources {
		if resource.cooldown == 0 {
			count++
		}
	}
	return count
}

func (e *flatlandEpisode) step(move float64) (int, bool, bool, bool, string) {
	move = clamp(move, -1, 1)
	moveStep := 0
	if move > 0.33 {
		moveStep = 1
	} else if move < -0.33 {
		moveStep = -1
	}

	wallCollision := false
	if moveStep != 0 {
		candidate := wrapFlatlandPosition(e.position + moveStep)
		if e.isWall(candidate) {
			wallCollision = true
			e.wallCollisions++
			e.rewardAcc -= flatlandWallPenalty
			e.energy -= flatlandWallEnergyPenalty
		} else {
			e.position = candidate
		}
	}

	hitFood := e.consumeFoodAtPosition()
	hitPoison := e.consumePoisonAtPosition()

	e.energy -= flatlandBaseMetabolic + flatlandMoveMetabolic*math.Abs(float64(moveStep))
	if moveStep == 0 {
		e.energy -= flatlandIdleMetabolic
	}
	if e.energy < 0 {
		e.energy = 0
	}
	e.rewardAcc += flatlandSurvivalReward
	e.age++

	if e.energy <= 0 {
		return moveStep, hitFood, hitPoison, wallCollision, "depleted"
	}
	if e.foodCollected >= e.forageGoal {
		return moveStep, hitFood, hitPoison, wallCollision, "forage_goal"
	}
	if e.age >= e.maxAge {
		return moveStep, hitFood, hitPoison, wallCollision, "age_limit"
	}
	return moveStep, hitFood, hitPoison, wallCollision, ""
}

func (e *flatlandEpisode) consumeFoodAtPosition() bool {
	for i := range e.food {
		resource := &e.food[i]
		if resource.cooldown > 0 || resource.position != e.position {
			continue
		}
		potency := clamp(resource.potency, flatlandFoodEnergyMin, flatlandFoodEnergyMax)
		resource.cooldown = flatlandFoodRespawn
		resource.potency = flatlandFoodEnergyMin
		e.energy += potency
		if e.energy > flatlandEnergyCap {
			e.energy = flatlandEnergyCap
		}
		e.foodCollected++
		e.rewardAcc += flatlandFoodReward * (potency / flatlandFoodEnergyMax)
		return true
	}
	return false
}

func (e *flatlandEpisode) consumePoisonAtPosition() bool {
	for i := range e.poison {
		resource := &e.poison[i]
		if resource.cooldown > 0 || resource.position != e.position {
			continue
		}
		potency := clamp(resource.potency, flatlandPoisonDamageMin, flatlandPoisonDamageMax)
		resource.cooldown = flatlandPoisonRespawn
		resource.potency = flatlandPoisonDamageMin
		e.energy -= potency
		e.poisonHits++
		e.rewardAcc -= flatlandPoisonPenalty * (potency / flatlandPoisonDamageMax)
		return true
	}
	return false
}

func (e *flatlandEpisode) isWall(position int) bool {
	_, blocked := e.walls[position]
	return blocked
}

func wrapFlatlandPosition(position int) int {
	wrapped := position % flatlandWorldSize
	if wrapped < 0 {
		wrapped += flatlandWorldSize
	}
	return wrapped
}

func signedRingDistance(from, to, size int) int {
	if size <= 0 {
		return 0
	}
	delta := (to - from) % size
	half := size / 2
	if delta > half {
		delta -= size
	}
	if delta < -half {
		delta += size
	}
	return delta
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

type flatlandIOBindings struct {
	distanceSetter        protoio.ScalarSensorSetter
	energySetter          protoio.ScalarSensorSetter
	poisonSetter          protoio.ScalarSensorSetter
	wallSetter            protoio.ScalarSensorSetter
	foodProximitySetter   protoio.ScalarSensorSetter
	poisonProximitySetter protoio.ScalarSensorSetter
	wallProximitySetter   protoio.ScalarSensorSetter
	resourceBalanceSetter protoio.ScalarSensorSetter
	distanceScanSetters   [flatlandScannerDensity]protoio.ScalarSensorSetter
	colorScanSetters      [flatlandScannerDensity]protoio.ScalarSensorSetter
	energyScanSetters     [flatlandScannerDensity]protoio.ScalarSensorSetter
	moveOutput            protoio.SnapshotActuator
	controlSurface        string
}

func flatlandIO(agent TickAgent) (flatlandIOBindings, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return flatlandIOBindings{}, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	distanceSetter, hasDistance, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandDistanceSensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	energySetter, hasEnergy, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandEnergySensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	poisonSetter, hasPoison, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandPoisonSensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	wallSetter, hasWall, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandWallSensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	foodProximitySetter, hasFoodProximity, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandFoodProximitySensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	poisonProximitySetter, hasPoisonProximity, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandPoisonProximitySensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	wallProximitySetter, hasWallProximity, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandWallProximitySensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	resourceBalanceSetter, hasResourceBalance, err := resolveOptionalFlatlandSetter(typed, protoio.FlatlandResourceBalanceSensorName)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	distanceScanSetters, distanceScanCount, err := resolveOptionalFlatlandScannerSetters(typed, flatlandDistanceScannerSensors)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	colorScanSetters, colorScanCount, err := resolveOptionalFlatlandScannerSetters(typed, flatlandColorScannerSensors)
	if err != nil {
		return flatlandIOBindings{}, err
	}
	energyScanSetters, energyScanCount, err := resolveOptionalFlatlandScannerSetters(typed, flatlandEnergyScannerSensors)
	if err != nil {
		return flatlandIOBindings{}, err
	}

	if distanceScanCount != 0 && distanceScanCount != flatlandScannerDensity {
		return flatlandIOBindings{}, fmt.Errorf(
			"agent %s has partial flatland distance scanner set (%d/%d); expected all or none",
			agent.ID(),
			distanceScanCount,
			flatlandScannerDensity,
		)
	}
	if colorScanCount != 0 && colorScanCount != flatlandScannerDensity {
		return flatlandIOBindings{}, fmt.Errorf(
			"agent %s has partial flatland color scanner set (%d/%d); expected all or none",
			agent.ID(),
			colorScanCount,
			flatlandScannerDensity,
		)
	}
	if energyScanCount != 0 && energyScanCount != flatlandScannerDensity {
		return flatlandIOBindings{}, fmt.Errorf(
			"agent %s has partial flatland energy scanner set (%d/%d); expected all or none",
			agent.ID(),
			energyScanCount,
			flatlandScannerDensity,
		)
	}
	if !hasDistance &&
		!hasEnergy &&
		!hasPoison &&
		!hasWall &&
		!hasFoodProximity &&
		!hasPoisonProximity &&
		!hasWallProximity &&
		!hasResourceBalance &&
		distanceScanCount == 0 &&
		colorScanCount == 0 &&
		energyScanCount == 0 {
		return flatlandIOBindings{}, fmt.Errorf(
			"agent %s missing flatland sensing surface; expected one or more flatland sensors",
			agent.ID(),
		)
	}

	actuatorName := protoio.FlatlandTwoWheelsActuatorName
	actuator, ok := typed.RegisteredActuator(actuatorName)
	if !ok {
		actuatorName = protoio.FlatlandMoveActuatorName
		actuator, ok = typed.RegisteredActuator(actuatorName)
		if !ok {
			return flatlandIOBindings{}, fmt.Errorf(
				"agent %s missing flatland actuator surface; expected %s or %s",
				agent.ID(),
				protoio.FlatlandTwoWheelsActuatorName,
				protoio.FlatlandMoveActuatorName,
			)
		}
	}
	moveOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return flatlandIOBindings{}, fmt.Errorf("actuator %s does not support output snapshot", actuatorName)
	}

	return flatlandIOBindings{
		distanceSetter:        distanceSetter,
		energySetter:          energySetter,
		poisonSetter:          poisonSetter,
		wallSetter:            wallSetter,
		foodProximitySetter:   foodProximitySetter,
		poisonProximitySetter: poisonProximitySetter,
		wallProximitySetter:   wallProximitySetter,
		resourceBalanceSetter: resourceBalanceSetter,
		distanceScanSetters:   distanceScanSetters,
		colorScanSetters:      colorScanSetters,
		energyScanSetters:     energyScanSetters,
		moveOutput:            moveOutput,
		controlSurface:        actuatorName,
	}, nil
}

func resolveOptionalFlatlandSetter(
	typed interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
	},
	sensorID string,
) (protoio.ScalarSensorSetter, bool, error) {
	sensor, ok := typed.RegisteredSensor(sensorID)
	if !ok {
		return nil, false, nil
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, false, fmt.Errorf("sensor %s does not support scalar set", sensorID)
	}
	return setter, true, nil
}

func resolveOptionalFlatlandScannerSetters(
	typed interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
	},
	sensorIDs [flatlandScannerDensity]string,
) ([flatlandScannerDensity]protoio.ScalarSensorSetter, int, error) {
	var setters [flatlandScannerDensity]protoio.ScalarSensorSetter
	count := 0
	for i, sensorID := range sensorIDs {
		setter, ok, err := resolveOptionalFlatlandSetter(typed, sensorID)
		if err != nil {
			return setters, 0, err
		}
		if ok {
			setters[i] = setter
			count++
		}
	}
	return setters, count, nil
}

func flatlandControlFromOutput(values []float64) (flatlandControl, error) {
	switch len(values) {
	case 0:
		return flatlandControl{}, fmt.Errorf("flatland requires at least one control output, got 0")
	case 1:
		return flatlandControl{
			move:  clamp(values[0], -1, 1),
			width: 1,
		}, nil
	default:
		left := clamp(values[0], -1, 1)
		right := clamp(values[1], -1, 1)
		avgDrive := 0.5 * (left + right)
		differential := right - left
		// Blend average drive with differential intent for stable 1D surrogate control.
		move := clamp(0.65*avgDrive+0.35*differential, -1, 1)
		return flatlandControl{
			move:  move,
			width: 2,
		}, nil
	}
}

func meanFlatlandScan(values [flatlandScannerDensity]float64) float64 {
	if flatlandScannerDensity <= 0 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(flatlandScannerDensity)
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
