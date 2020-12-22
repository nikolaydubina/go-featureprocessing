package transformers_test

import (
	"testing"

	. "github.com/nikolaydubina/go-featureprocessing/transformers"
	"github.com/stretchr/testify/assert"
)

func TestKBinsDiscretizerTransform(t *testing.T) {
	samples := []struct {
		name      string
		n         int
		quantiles []float64
		input     float64
		output    float64
	}{
		{"basic1", 4, []float64{25, 50, 75, 100}, 0, 1},
		{"basic2", 4, []float64{25, 50, 75, 100}, 11, 1},
		{"basic3", 4, []float64{25, 50, 75, 100}, 25, 1},
		{"basic4", 4, []float64{25, 50, 75, 100}, 40, 2},
		{"basic5", 4, []float64{25, 50, 75, 100}, 50, 2},
		{"basic6", 4, []float64{25, 50, 75, 100}, 80, 4},
		{"above_max", 4, []float64{25, 50, 75, 100}, 101, 5},
		{"empty", 0, nil, 10, 0},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := KBinsDiscretizer{QuantileScaler{NQuantiles: s.n, Quantiles: s.quantiles}}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)
		})
	}
}

func TestKBinsDiscretizerTransformFit(t *testing.T) {
	samples := []struct {
		name      string
		n         int
		quantiles []float64
		vals      []float64
	}{
		{"noinput", 1000, nil, nil},
		{"basic", 4, []float64{25, 50, 75, 100}, []float64{25, 50, 75, 100}},
		{"reverse_order", 4, []float64{25, 50, 75, 100}, []float64{100, 75, 50, 25}},
		{"negative", 4, []float64{-100, -75, -50, -25}, []float64{-25, -50, -75, -100}},
		{"one_element", 1, []float64{10}, []float64{10}},
		{"less_elements_than_quantiles", 3, []float64{1, 2, 3}, []float64{1, 2, 3}},
		{"less_elements_than_quantiles_negative", 3, []float64{-3, -2, -1}, []float64{-1, -3, -2}},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := KBinsDiscretizer{QuantileScaler{NQuantiles: s.n}}
			encoder.Fit(s.vals)
			assert.Equal(t, KBinsDiscretizer{QuantileScaler{NQuantiles: s.n, Quantiles: s.quantiles}}, encoder)
		})
	}

	t.Run("nquantiles is larger than num input vals", func(t *testing.T) {
		encoder := KBinsDiscretizer{QuantileScaler{NQuantiles: 10}}
		encoder.Fit([]float64{1, 2, 3})
		assert.Equal(t, KBinsDiscretizer{QuantileScaler{NQuantiles: 3, Quantiles: []float64{1, 2, 3}}}, encoder)
	})

	t.Run("nquantiles is zero in beginning", func(t *testing.T) {
		encoder := KBinsDiscretizer{QuantileScaler{}}
		encoder.Fit(nil)
		assert.Equal(t, KBinsDiscretizer{QuantileScaler{NQuantiles: 1000}}, encoder)
	})
}
