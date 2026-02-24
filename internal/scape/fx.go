package scape

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type FXScape struct{}

type fxSeries struct {
	name   string
	values []float64
}

var (
	fxSeriesSourceMu sync.RWMutex
	fxSeriesSource   = defaultFXSeries()
)

func (FXScape) Name() string {
	return "fx"
}

func (FXScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return FXScape{}.EvaluateMode(ctx, agent, "gt")
}

func (FXScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := fxConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateFXWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateFXWithStep(ctx, runner, cfg)
}

func evaluateFXWithStep(ctx context.Context, runner StepAgent, cfg fxModeConfig) (Fitness, Trace, error) {
	return evaluateFX(ctx, cfg, func(ctx context.Context, percept []float64) (float64, error) {
		out, err := runner.RunStep(ctx, percept)
		if err != nil {
			return 0, err
		}
		if len(out) != 1 {
			return 0, fmt.Errorf("fx requires one output, got %d", len(out))
		}
		return out[0], nil
	})
}

func evaluateFXWithTick(ctx context.Context, ticker TickAgent, cfg fxModeConfig) (Fitness, Trace, error) {
	io, err := fxIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateFX(ctx, cfg, func(ctx context.Context, percept []float64) (float64, error) {
		if len(percept) < 2 {
			return 0, fmt.Errorf("fx percept width <2 for tick agent: %d", len(percept))
		}
		io.price.Set(percept[0])
		io.signal.Set(percept[1])
		if io.momentum != nil && len(percept) > 2 {
			io.momentum.Set(percept[2])
		}
		if io.volatility != nil && len(percept) > 4 {
			io.volatility.Set(percept[4])
		}
		if io.nav != nil && len(percept) > 7 {
			io.nav.Set(percept[7])
		}
		if io.drawdown != nil && len(percept) > 6 {
			io.drawdown.Set(percept[6])
		}
		if io.position != nil && len(percept) > 8 {
			io.position.Set(percept[8])
		}
		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, err
		}
		lastOutput := io.tradeOutput.Last()
		if len(lastOutput) > 0 {
			return lastOutput[0], nil
		}
		if len(out) > 0 {
			return out[0], nil
		}
		return 0, nil
	})
}

func evaluateFX(
	ctx context.Context,
	cfg fxModeConfig,
	chooseTrade func(context.Context, []float64) (float64, error),
) (Fitness, Trace, error) {
	series := currentFXSeries()
	account := newFXAccount()
	ordersOpened := 0
	ordersClosed := 0
	directionChanges := 0
	marginCall := false
	turnover := 0.0
	lastAction := 0.0
	lastQuote := fxPrice(series, cfg.startStep)
	prevQuote := fxPrice(series, maxIntFX(0, cfg.startStep-1))

	for i := 0; i < cfg.steps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		step := cfg.startStep + i
		quote := fxPrice(series, step)
		signal := fxSignal(series, step)
		lastQuote = quote

		updateFXMarkToMarket(&account, quote)

		momentum1 := 0.0
		if prevQuote != 0 {
			momentum1 = (quote - prevQuote) / prevQuote
		}
		backQuote := prevQuote
		if step >= 4 {
			backQuote = fxPrice(series, step-4)
		}
		momentum4 := 0.0
		if backQuote != 0 {
			momentum4 = (quote - backQuote) / backQuote
		}
		volatility := math.Abs(momentum1 - momentum4)
		percept := fxPerceptVector(quote, signal, momentum1, momentum4, volatility, account, lastAction)

		rawAction, err := chooseTrade(ctx, percept)
		if err != nil {
			return 0, nil, err
		}
		action := fxTradeAction(rawAction)

		if action != lastAction {
			turnover += math.Abs(action - lastAction)
			if action != 0 && lastAction != 0 {
				directionChanges++
			}
		}
		opened, closed := applyFXTradeAction(&account, quote, action)
		ordersOpened += opened
		ordersClosed += closed
		lastAction = action
		prevQuote = quote

		updateFXMarkToMarket(&account, quote)
		if account.netAssetValue <= fxMarginCallFloor {
			marginCall = true
			break
		}
	}

	if !marginCall && account.order != nil {
		if closeFXOrder(&account, lastQuote) {
			ordersClosed++
		}
	}
	updateFXMarkToMarket(&account, lastQuote)

	netWorth := account.netAssetValue
	returnPct := (netWorth - fxInitialBalance) / fxInitialBalance
	drawdownRatio := account.maxDrawdown / fxInitialBalance
	tradeCount := float64(ordersOpened + ordersClosed)

	fitnessRaw := returnPct*2.6 - drawdownRatio*1.4 - 0.002*tradeCount
	if ordersOpened == 0 {
		fitnessRaw -= 0.08
	}
	if marginCall {
		fitnessRaw -= 0.35
	}
	fitness := 1.0 / (1.0 + math.Exp(-fitnessRaw))
	if math.IsNaN(fitness) || math.IsInf(fitness, 0) {
		fitness = 0
	}

	position := 0.0
	entry := 0.0
	units := 0.0
	if account.order != nil {
		position = account.order.position
		entry = account.order.entry
		units = account.order.units
	}

	return Fitness(fitness), Trace{
		"equity":            netWorth / fxInitialBalance,
		"turnover":          turnover,
		"mode":              cfg.mode,
		"steps":             cfg.steps,
		"start_step":        cfg.startStep,
		"series_name":       series.name,
		"series_points":     len(series.values),
		"balance":           account.balance,
		"net_worth":         netWorth,
		"realized_pl":       account.realizedPL,
		"unrealized_pl":     account.unrealizedPL,
		"max_drawdown":      account.maxDrawdown,
		"margin_call":       marginCall,
		"orders_opened":     ordersOpened,
		"orders_closed":     ordersClosed,
		"direction_changes": directionChanges,
		"position":          position,
		"entry":             entry,
		"units":             units,
		"feature_width":     fxPerceptWidth,
	}, nil
}

type fxOrder struct {
	position         float64
	entry            float64
	current          float64
	units            float64
	change           float64
	percentageChange float64
	profit           float64
}

type fxAccount struct {
	balance       float64
	realizedPL    float64
	unrealizedPL  float64
	netAssetValue float64
	order         *fxOrder
	maxNetWorth   float64
	maxDrawdown   float64
}

const (
	fxInitialBalance  = 300.0
	fxMarginCallFloor = 100.0
	fxSpread          = 0.000150
	fxLeverage        = 50.0
	fxOrderBudget     = 100.0
	fxTradeThreshold  = 0.33
	fxPerceptWidth    = 10
)

func newFXAccount() fxAccount {
	return fxAccount{
		balance:       fxInitialBalance,
		netAssetValue: fxInitialBalance,
		maxNetWorth:   fxInitialBalance,
	}
}

func updateFXMarkToMarket(account *fxAccount, quote float64) {
	if account.order == nil {
		account.unrealizedPL = 0
		account.netAssetValue = account.balance
	} else {
		order := account.order
		change := quote - order.entry
		pctChange := 0.0
		if order.entry != 0 {
			pctChange = (change / order.entry) * 100
		}
		profit := order.position * change * order.units
		order.current = quote
		order.change = change
		order.percentageChange = pctChange
		order.profit = profit
		account.unrealizedPL = profit
		account.netAssetValue = account.balance + account.unrealizedPL
	}

	if account.netAssetValue > account.maxNetWorth {
		account.maxNetWorth = account.netAssetValue
	}
	drawdown := account.maxNetWorth - account.netAssetValue
	if drawdown > account.maxDrawdown {
		account.maxDrawdown = drawdown
	}
}

func fxTradeAction(raw float64) float64 {
	if raw > fxTradeThreshold {
		return 1
	}
	if raw < -fxTradeThreshold {
		return -1
	}
	return 0
}

func applyFXTradeAction(account *fxAccount, quote, action float64) (opened int, closed int) {
	if account.order == nil {
		if action != 0 {
			if openFXOrder(account, quote, action) {
				opened = 1
			}
		}
		return opened, closed
	}

	if action == 0 {
		if closeFXOrder(account, quote) {
			closed = 1
		}
		return opened, closed
	}

	currentPosition := account.order.position
	if action == currentPosition {
		return opened, closed
	}

	if closeFXOrder(account, quote) {
		closed = 1
	}
	if openFXOrder(account, quote, action) {
		opened = 1
	}
	return opened, closed
}

func openFXOrder(account *fxAccount, quote, action float64) bool {
	entry := quote + fxSpread*action
	if entry <= 0 {
		entry = quote
	}
	units := math.Round((fxOrderBudget * fxLeverage) / math.Max(entry, 0.0001))
	if units < 1 {
		units = 1
	}

	account.order = &fxOrder{
		position: action,
		entry:    entry,
		current:  quote,
		units:    units,
	}
	updateFXMarkToMarket(account, quote)
	return true
}

func closeFXOrder(account *fxAccount, quote float64) bool {
	if account.order == nil {
		return false
	}

	updateFXMarkToMarket(account, quote)
	account.balance += account.unrealizedPL
	account.realizedPL += account.unrealizedPL
	account.unrealizedPL = 0
	account.order = nil
	account.netAssetValue = account.balance
	if account.netAssetValue > account.maxNetWorth {
		account.maxNetWorth = account.netAssetValue
	}
	drawdown := account.maxNetWorth - account.netAssetValue
	if drawdown > account.maxDrawdown {
		account.maxDrawdown = drawdown
	}
	return true
}

type fxIOBindings struct {
	price       protoio.ScalarSensorSetter
	signal      protoio.ScalarSensorSetter
	momentum    protoio.ScalarSensorSetter
	volatility  protoio.ScalarSensorSetter
	nav         protoio.ScalarSensorSetter
	drawdown    protoio.ScalarSensorSetter
	position    protoio.ScalarSensorSetter
	tradeOutput protoio.SnapshotActuator
}

func fxIO(agent TickAgent) (fxIOBindings, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return fxIOBindings{}, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	price, ok := typed.RegisteredSensor(protoio.FXPriceSensorName)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FXPriceSensorName)
	}
	priceSetter, ok := price.(protoio.ScalarSensorSetter)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("sensor %s does not support scalar set", protoio.FXPriceSensorName)
	}

	signal, ok := typed.RegisteredSensor(protoio.FXSignalSensorName)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FXSignalSensorName)
	}
	signalSetter, ok := signal.(protoio.ScalarSensorSetter)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("sensor %s does not support scalar set", protoio.FXSignalSensorName)
	}

	momentumSetter, err := optionalFXSensorSetter(typed, protoio.FXMomentumSensorName)
	if err != nil {
		return fxIOBindings{}, err
	}
	volatilitySetter, err := optionalFXSensorSetter(typed, protoio.FXVolatilitySensorName)
	if err != nil {
		return fxIOBindings{}, err
	}
	navSetter, err := optionalFXSensorSetter(typed, protoio.FXNAVSensorName)
	if err != nil {
		return fxIOBindings{}, err
	}
	drawdownSetter, err := optionalFXSensorSetter(typed, protoio.FXDrawdownSensorName)
	if err != nil {
		return fxIOBindings{}, err
	}
	positionSetter, err := optionalFXSensorSetter(typed, protoio.FXPositionSensorName)
	if err != nil {
		return fxIOBindings{}, err
	}

	actuator, ok := typed.RegisteredActuator(protoio.FXTradeActuatorName)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.FXTradeActuatorName)
	}
	tradeOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return fxIOBindings{}, fmt.Errorf("actuator %s does not support output snapshot", protoio.FXTradeActuatorName)
	}
	return fxIOBindings{
		price:       priceSetter,
		signal:      signalSetter,
		momentum:    momentumSetter,
		volatility:  volatilitySetter,
		nav:         navSetter,
		drawdown:    drawdownSetter,
		position:    positionSetter,
		tradeOutput: tradeOutput,
	}, nil
}

func optionalFXSensorSetter(
	typed interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
	},
	sensorID string,
) (protoio.ScalarSensorSetter, error) {
	sensor, ok := typed.RegisteredSensor(sensorID)
	if !ok {
		return nil, nil
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, fmt.Errorf("sensor %s does not support scalar set", sensorID)
	}
	return setter, nil
}

type fxModeConfig struct {
	mode      string
	steps     int
	startStep int
}

// ResetFXSeriesSource restores the deterministic built-in FX series.
func ResetFXSeriesSource() {
	fxSeriesSourceMu.Lock()
	defer fxSeriesSourceMu.Unlock()
	fxSeriesSource = defaultFXSeries()
}

// LoadFXSeriesCSV loads price points from CSV and makes the series active.
// The last non-empty column per row is interpreted as the price.
func LoadFXSeriesCSV(path string) error {
	series, err := loadFXSeriesCSV(path)
	if err != nil {
		return err
	}
	fxSeriesSourceMu.Lock()
	defer fxSeriesSourceMu.Unlock()
	fxSeriesSource = series
	return nil
}

func loadFXSeriesCSV(path string) (fxSeries, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return fxSeries{}, fmt.Errorf("fx csv path is required")
	}

	f, err := os.Open(path)
	if err != nil {
		return fxSeries{}, fmt.Errorf("open fx csv %s: %w", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	values := make([]float64, 0, 512)
	row := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fxSeries{}, fmt.Errorf("read fx csv row %d: %w", row+1, err)
		}
		row++

		field, ok := fxCSVValueField(record)
		if !ok {
			continue
		}
		price, err := strconv.ParseFloat(field, 64)
		if err != nil {
			if len(values) == 0 {
				continue
			}
			return fxSeries{}, fmt.Errorf("parse fx csv price row %d: %w", row, err)
		}
		if price <= 0 {
			return fxSeries{}, fmt.Errorf("invalid fx csv price row %d: %f", row, price)
		}
		values = append(values, price)
	}

	if len(values) < 8 {
		return fxSeries{}, fmt.Errorf("fx csv %s requires at least 8 price rows, got %d", path, len(values))
	}

	return fxSeries{
		name:   fmt.Sprintf("fx.csv.%s", filepath.Base(path)),
		values: values,
	}, nil
}

func fxCSVValueField(record []string) (string, bool) {
	for i := len(record) - 1; i >= 0; i-- {
		field := strings.TrimSpace(record[i])
		if field != "" {
			return field, true
		}
	}
	return "", false
}

func defaultFXSeries() fxSeries {
	const total = 512
	values := make([]float64, total)
	for i := 0; i < total; i++ {
		values[i] = fxSyntheticPrice(i)
	}
	return fxSeries{name: "fx.synthetic.v2", values: values}
}

func currentFXSeries() fxSeries {
	fxSeriesSourceMu.RLock()
	defer fxSeriesSourceMu.RUnlock()
	return fxSeriesSource
}

func fxConfigForMode(mode string) (fxModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return fxModeConfig{mode: "gt", steps: 64, startStep: 0}, nil
	case "validation":
		return fxModeConfig{mode: "validation", steps: 48, startStep: 128}, nil
	case "test":
		return fxModeConfig{mode: "test", steps: 48, startStep: 256}, nil
	case "benchmark":
		return fxModeConfig{mode: "benchmark", steps: 48, startStep: 256}, nil
	default:
		return fxModeConfig{}, fmt.Errorf("unsupported fx mode: %s", mode)
	}
}

func fxSyntheticPrice(step int) float64 {
	t := float64(step)
	base := 1.05 + 0.0006*t
	cycle := 0.035*math.Sin(t*0.11) + 0.012*math.Sin(t*0.37+0.9)
	regime := 0.0
	switch {
	case step >= 80 && step < 160:
		regime = 0.0008 * float64(step-80)
	case step >= 160 && step < 260:
		regime = 0.064 - 0.0009*float64(step-160)
	case step >= 260:
		regime = -0.026 + 0.00035*float64(step-260)
	}
	price := base + cycle + regime
	if price < 0.2 {
		return 0.2
	}
	return price
}

func fxPrice(series fxSeries, step int) float64 {
	if len(series.values) == 0 {
		return fxSyntheticPrice(step)
	}
	if step < 0 {
		step = 0
	}
	if step >= len(series.values) {
		step = len(series.values) - 1
	}
	return clampFX(series.values[step], 0.2, math.MaxFloat64)
}

func fxSignal(series fxSeries, step int) float64 {
	if step <= 0 {
		return 0
	}
	p0 := fxPrice(series, step)
	p1 := fxPrice(series, step-1)
	mom1 := (p0 - p1) / p1
	if step < 4 {
		return clampFX(mom1, -1, 1)
	}
	p4 := fxPrice(series, step-4)
	mom4 := (p0 - p4) / p4
	vol := math.Abs(mom1 - mom4)
	signal := 0.65*mom1 + 0.35*mom4 - 0.2*vol
	return clampFX(signal, -1, 1)
}

func fxPerceptVector(
	quote, signal, momentum1, momentum4, volatility float64,
	account fxAccount,
	lastAction float64,
) []float64 {
	position := 0.0
	exposure := 0.0
	if account.order != nil {
		position = account.order.position
		exposure = (account.order.units * quote) / fxInitialBalance
	}

	return []float64{
		quote,
		signal,
		momentum1,
		momentum4,
		volatility,
		account.unrealizedPL / fxInitialBalance,
		account.maxDrawdown / fxInitialBalance,
		account.netAssetValue / fxInitialBalance,
		position,
		0.2*lastAction + 0.01*exposure,
	}
}

func maxIntFX(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clampFX(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
