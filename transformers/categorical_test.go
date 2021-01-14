package transformers_test

import (
	"testing"

	. "github.com/nikolaydubina/go-featureprocessing/transformers"
	"github.com/stretchr/testify/assert"
)

func TestOneHotEncoderFit(t *testing.T) {
	samples := []struct {
		name   string
		input  []string
		output []string
		n      int
	}{
		{"basic", []string{"a", "b", "a", "a", "a"}, []string{"a", "b"}, 2},
		{"empty", []string{}, []string{}, 0},
		{"nil", nil, nil, 0},
		{"same_string", []string{"a", "a", "a"}, []string{"a"}, 1},
		{"empty_string", []string{"", "", ""}, []string{""}, 1},
		{"zeros_single", []string{""}, []string{""}, 1},
		{"single", []string{"a"}, []string{"a"}, 1},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OneHotEncoder{}
			encoder.Fit(s.input)
			assert.Equal(t, OneHotEncoder{Values: s.output}, encoder)
			assert.Equal(t, s.n, encoder.NumFeatures())
		})
	}
}

func TestOneHotEncoderTransform(t *testing.T) {
	samples := []struct {
		name   string
		vals   []string
		input  string
		output []float64
	}{
		{"basic", []string{"a", "b"}, "a", []float64{1, 0}},
		{"basic", []string{"a", "b"}, "b", []float64{0, 1}},
		{"none", []string{"a", "b"}, "c", []float64{0, 0}},
		{"empty_input", []string{"a", "b"}, "", []float64{0, 0}},
		{"empty_vals", []string{}, "a", nil},
		{"nil_vals", nil, "a", nil},
		{"zeros_single", []string{""}, "", []float64{1}},
		{"single", []string{"a"}, "a", []float64{1}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OneHotEncoder{Values: s.vals}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})

		if len(s.output) > 0 {
			t.Run(s.name+"_inplace", func(t *testing.T) {
				encoder := OneHotEncoder{Values: s.vals}
				assert.Equal(t, s.output, encoder.Transform(s.input))

				features := make([]float64, encoder.NumFeatures())
				encoder.TransformInplace(features, s.input)
				assert.Equal(t, s.output, features)

				features = make([]float64, encoder.NumFeatures()+100)
				features[0] = 11223344556677
				features[1] = 10101010110101
				features[10] = 12341231 // has to overwrite this
				features[99] = 12312312312312

				expected := make([]float64, len(features))
				copy(expected, features)
				copy(expected[10:], s.output)

				encoder.TransformInplace(features[10:10+encoder.NumFeatures()], s.input)
				assert.Equal(t, expected, features)
			})
		}
	}

	t.Run("inplace does not compute when input is wrong", func(t *testing.T) {
		encoder := OneHotEncoder{Values: []string{"a", "b"}}

		features := []float64{1.1, 2.1, 3.1, 4.1}
		encoder.TransformInplace(features, "a")
		assert.Equal(t, []float64{1.1, 2.1, 3.1, 4.1}, features)
	})
}

func TestOneHotEncoderFeatureNames(t *testing.T) {
	t.Run("feature names on empty transformer", func(t *testing.T) {
		var encoder *OneHotEncoder
		assert.Equal(t, []string(nil), encoder.FeatureNames())
	})

	t.Run("feature names", func(t *testing.T) {
		encoder := OneHotEncoder{Values: []string{"a", "b"}}
		assert.Equal(t, []string{"a", "b"}, encoder.FeatureNames())
	})
}

func TestOrdinalEncoderFit(t *testing.T) {
	samples := []struct {
		name   string
		input  []string
		output map[string]float64
	}{
		{"basic", []string{"a", "b", "a", "a", "a"}, map[string]float64{"a": 1, "b": 2}},
		{"empty", []string{}, nil},
		{"nil", nil, nil},
		{"same_string", []string{"a", "a", "a"}, map[string]float64{"a": 1}},
		{"empty_string", []string{"", "", ""}, map[string]float64{"": 1}},
		{"zeros_single", []string{""}, map[string]float64{"": 1}},
		{"single", []string{"a"}, map[string]float64{"a": 1}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OrdinalEncoder{}
			encoder.Fit(s.input)
			assert.Equal(t, OrdinalEncoder{Mapping: s.output}, encoder)
		})
	}
}

func TestOrdinalEncoderTransform(t *testing.T) {
	samples := []struct {
		name   string
		vals   map[string]float64
		input  string
		output float64
	}{
		{"basic", map[string]float64{"a": 1, "b": 3}, "a", 1},
		{"basic", map[string]float64{"a": 1, "b": 3}, "b", 3},
		{"none", map[string]float64{"a": 1, "b": 3}, "c", 0},
		{"empty_input", map[string]float64{"a": 1, "b": 3}, "", 0},
		{"empty_vals", map[string]float64{}, "a", 0},
		{"nil_vals", nil, "a", 0},
		{"zero_single", map[string]float64{"": 1}, "", 1},
		{"single", map[string]float64{"a": 1}, "a", 1},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OrdinalEncoder{Mapping: s.vals}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})
	}
}
