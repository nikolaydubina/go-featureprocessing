package transformers_test

import (
	"testing"

	. "github.com/nikolaydubina/go-featureprocessing/transformers"
	"github.com/stretchr/testify/assert"
)

func TestSampleNormalizserL1(t *testing.T) {
	samples := []struct {
		name   string
		input  []float64
		output []float64
	}{
		{"basic", []float64{1, 2, 3, 4}, []float64{0.1, 0.2, 0.3, 0.4}},
		{"empty", []float64{}, []float64{}},
		{"nil", nil, nil},
		{"zeros", []float64{0, 0, 0}, []float64{0, 0, 0}},
		{"zeros_single", []float64{0}, []float64{0}},
		{"single", []float64{5}, []float64{1}},
		{"single_negative", []float64{-5}, []float64{-1}},
		{"negative", []float64{1, 2, 3, -4}, []float64{0.1, 0.2, 0.3, -0.4}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := SampleNormalizerL1{}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)
		})
	}
}

func TestSampleNormalizserL2(t *testing.T) {
	samples := []struct {
		name   string
		input  []float64
		output []float64
	}{
		{"basic", []float64{1, 1, 3, 5, 8}, []float64{0.1, 0.1, 0.3, 0.5, 0.8}},
		{"empty", []float64{}, []float64{}},
		{"nil", nil, nil},
		{"zeros", []float64{0, 0, 0}, []float64{0, 0, 0}},
		{"zeros_single", []float64{0}, []float64{0}},
		{"single", []float64{5}, []float64{1}},
		{"single_negative", []float64{-5}, []float64{-1}},
		{"basic", []float64{1, 1, -3, 5, -8}, []float64{0.1, 0.1, -0.3, 0.5, -0.8}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := SampleNormalizerL2{}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)
		})
	}
}
