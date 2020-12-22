package transformers

import "sort"

// KBinsDiscretizer based on quantile strategy
type KBinsDiscretizer struct {
	QuantileScaler
}

// Fit fits quantile scaler
func (t *KBinsDiscretizer) Fit(vals []float64) {
	t.QuantileScaler.Fit(vals)
}

// Transform finds index of matched quantile for input
func (t *KBinsDiscretizer) Transform(v float64) float64 {
	if len(t.QuantileScaler.Quantiles) == 0 {
		return 0
	}
	i := sort.SearchFloat64s(t.Quantiles[:], v)
	if i >= t.NQuantiles {
		return float64(t.NQuantiles) + 1
	}
	return float64(i) + 1
}
