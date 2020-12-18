package transformers_test

import (
	"testing"

	. "github.com/nikolaydubina/go-featureprocessing/transformers"
	"github.com/stretchr/testify/assert"
)

func TestCountVectorizerFit(t *testing.T) {
	samples := []struct {
		name   string
		input  []string
		output map[string]int
	}{
		{"basic", []string{"a b", "b a", "a", "b", ""}, map[string]int{"a": 0, "b": 1}},
		{"same_string", []string{"a", "a", "a"}, map[string]int{"a": 0}},
		{"empty_string", []string{"", "", ""}, map[string]int{}},
		{"zeros_single", []string{""}, map[string]int{}},
		{"single", []string{"a"}, map[string]int{"a": 0}},
		{"empty", nil, nil},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := CountVectorizer{}
			encoder.Fit(s.input)
			assert.Equal(t, CountVectorizer{Mapping: s.output, Separator: " "}, encoder)
		})
	}

	t.Run("num features is zero for nil encoder", func(t *testing.T) {
		var encoder *CountVectorizer
		assert.Equal(t, 0, encoder.NumFeatures())
	})

	t.Run("transform returns nil on nil encoder", func(t *testing.T) {
		var encoder *CountVectorizer
		assert.Equal(t, []float64(nil), encoder.Transform("asdf"))
	})
}

func TestTfIdfVectorizerFit(t *testing.T) {
	samples := []struct {
		name        string
		ndocs       int
		doccount    map[int]int
		mapping     map[string]int
		input       []string
		numFeatures int
	}{
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, []string{"a a a b b", "a a a c", "a a", "a a a", "a a a a", "a a a c c"}, 3},
		{"empty encoder empty input", 0, map[int]int(nil), map[string]int(nil), nil, 0},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := TfIdfVectorizer{}
			expectedEncoder := TfIdfVectorizer{
				CountVectorizer: CountVectorizer{Mapping: s.mapping, Separator: " "},
				NumDocuments:    s.ndocs,
				DocCount:        s.doccount,
			}
			encoder.Fit(s.input)
			assert.Equal(t, expectedEncoder, encoder)
			assert.Equal(t, s.numFeatures, encoder.NumFeatures())
		})
	}

	t.Run("transofmer is nil", func(t *testing.T) {
		var encoder *TfIdfVectorizer
		assert.Equal(t, []float64(nil), encoder.Transform("asdf asdf"))
		assert.Equal(t, 0, encoder.NumFeatures())
	})
}

// test is based on data from: https://scikit-learn.org/stable/modules/feature_extraction.html
func TestTfIdfVectorizerTransform(t *testing.T) {
	samples := []struct {
		name     string
		ndocs    int
		doccount map[int]int
		mapping  map[string]int
		input    string
		output   []float64
	}{
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a a c", []float64{0.8194099510753755, 0, 0.5732079309279058}},
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a", []float64{1, 0, 0}},
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a a", []float64{1, 0, 0}},
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a a a", []float64{1, 0, 0}},
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a a b b", []float64{0.47330339145578754, 0.8808994832762984, 0}},
		{"basic", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "a a a c c", []float64{0.58149260706886, 0, 0.8135516873095773}},
		{"not found", 6, map[int]int{0: 6, 1: 1, 2: 2}, map[string]int{"a": 0, "b": 1, "c": 2}, "dddd", []float64{0, 0, 0}},
		{"empty input", 2, map[int]int{0: 1, 1: 2}, map[string]int{"a": 0, "b": 1}, "     ", []float64{0, 0}},
		{"empty vals", 2, map[int]int{0: 1, 1: 2}, map[string]int{}, " b  a  ", []float64{}},
		{"nil input", 2, map[int]int{0: 1, 1: 2}, map[string]int{}, "", []float64{}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := TfIdfVectorizer{
				CountVectorizer: CountVectorizer{Mapping: s.mapping, Separator: " "},
				NumDocuments:    s.ndocs,
				DocCount:        s.doccount,
			}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})
	}
}
