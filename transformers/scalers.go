package transformers

import (
	"math"
	"sort"
)

type Identity struct{}

func (t *Identity) Fit(_ []float64) {}

func (t *Identity) Transform(v float64) float64 {
	return v
}

type MinMaxScaler struct {
	Min float64
	Max float64
}

func (t *MinMaxScaler) Fit(vals []float64) {
	for i, v := range vals {
		if i == 0 {
			t.Min = v
			t.Max = v
		}
		if v < t.Min {
			t.Min = v
		}
		if v > t.Max {
			t.Max = v
		}
	}
}

func (t *MinMaxScaler) Transform(v float64) float64 {
	if t.Min == t.Max {
		return 0
	}
	if v < t.Min {
		return 0.
	}
	if v > t.Max {
		return 1.
	}
	return (v - t.Min) / (t.Max - t.Min)
}

type MaxAbsScaler struct {
	Max float64
}

func (t *MaxAbsScaler) Fit(vals []float64) {
	for i, v := range vals {
		if i == 0 {
			t.Max = v
		}
		if math.Abs(v) > t.Max {
			t.Max = math.Abs(v)
		}
	}
}

func (t *MaxAbsScaler) Transform(v float64) float64 {
	if t.Max == 0 {
		return 0
	}
	if v > math.Abs(t.Max) {
		return 1.
	}
	if v < -math.Abs(t.Max) {
		return -1.
	}
	return v / math.Abs(t.Max)
}

// StandardScaler transforms feature into normal standard distribution.
type StandardScaler struct {
	Mean float64
	STD  float64
}

func (t *StandardScaler) Fit(vals []float64) {
	sum := 0.
	for _, v := range vals {
		sum += v
	}
	if len(vals) > 0 {
		t.Mean = sum / float64(len(vals))
		t.STD = std(vals, t.Mean)
	}
}

func (t *StandardScaler) Transform(v float64) float64 {
	return (v - t.Mean) / t.STD
}

// QuantileScaler transforms any distribution to uniform distribution
// This is done by mapping values to quantiles they belong to
type QuantileScaler struct {
	Quantiles  []float64
	NQuantiles int // used only for fitting
}

// Fit sets parameters for qunatiles based on input.
// If input is smaller than number of quantiles, then using length of input.
func (t *QuantileScaler) Fit(vals []float64) {
	if t.NQuantiles == 0 {
		t.NQuantiles = 1000
	}
	if len(vals) == 0 {
		return
	}
	if len(vals) < t.NQuantiles {
		t.NQuantiles = len(vals)
	}

	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)

	if len(t.Quantiles) != t.NQuantiles {
		t.Quantiles = make([]float64, t.NQuantiles)
	}
	f := float64(len(sorted)) / float64(t.NQuantiles)
	for i, _ := range t.Quantiles {
		idx := int(float64(i) * f)
		t.Quantiles[i] = sorted[idx]
	}
	return
}

func (t *QuantileScaler) Transform(v float64) float64 {
	if t == nil || len(t.Quantiles) == 0 {
		return 0
	}
	i := sort.SearchFloat64s(t.Quantiles[:], v)
	if i >= len(t.Quantiles) {
		return 1.
	}
	return float64(i+1) / float64(len(t.Quantiles))
}
