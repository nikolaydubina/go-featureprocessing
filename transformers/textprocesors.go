package transformers

import (
	"math"
	"strings"
)

// CountVectorizer performs bag of words encoding of text.
type CountVectorizer struct {
	Mapping   map[string]uint
	Separator string // default space
}

// Fit assigns a number from 0 to N for each word in input, where N is number of words
func (t *CountVectorizer) Fit(vals []string) {
	if t.Separator == "" {
		t.Separator = " "
	}
	if len(vals) == 0 {
		return
	}
	t.Mapping = make(map[string]uint)
	var count uint = 0
	for _, v := range vals {
		ws := strings.Split(v, t.Separator)
		for _, w := range ws {
			if w == "" {
				continue
			}
			if _, ok := t.Mapping[w]; !ok {
				t.Mapping[w] = count
				count++
			}
		}
	}
}

// NumFeatures returns num of features made for single input field
func (t *CountVectorizer) NumFeatures() int {
	if t == nil {
		return 0
	}
	return len(t.Mapping)
}

// Transform counts how many time each word appeared in input
func (t *CountVectorizer) Transform(v string) []float64 {
	if t == nil {
		return nil
	}
	if len(t.Mapping) == 0 {
		return nil
	}
	counts := make([]float64, t.NumFeatures())

	// sanity check, do not transform on invalid transformer
	numf := uint(t.NumFeatures())
	for _, idx := range t.Mapping {
		if idx >= numf {
			return counts
		}
	}

	for _, w := range strings.Split(v, t.Separator) {
		if i, ok := t.Mapping[w]; ok {
			counts[i]++
		}
	}
	return counts
}

// TFIDFVectorizer performs tf-idf vectorization on top of count vectorization.
// Based on: https://scikit-learn.org/stable/modules/feature_extraction.html
// Using non-smooth version, adding 1 to log instead of denominator in idf.
type TFIDFVectorizer struct {
	CountVectorizer
	DocCount     map[int]uint // number of documents word appeared in
	NumDocuments int
	Normalizer   SampleNormalizerL2
}

// Fit fits CountVectorizer and extra information for tf-idf computation
func (t *TFIDFVectorizer) Fit(vals []string) {
	t.CountVectorizer.Fit(vals)
	if len(vals) == 0 {
		return
	}

	t.NumDocuments = len(vals)
	t.DocCount = make(map[int]uint)

	// second pass over whole input to count how many documents each word appeared in
	for _, v := range vals {
		counts := t.CountVectorizer.Transform(v)
		for i, v := range counts {
			if v > 0 {
				t.DocCount[i]++
			}
		}
	}
}

// NumFeatures returns number of features for single field
func (t *TFIDFVectorizer) NumFeatures() int {
	if t == nil {
		return 0
	}
	return len(t.CountVectorizer.Mapping)
}

// Transform performs tf-idf computation
func (t *TFIDFVectorizer) Transform(v string) []float64 {
	if t == nil {
		return nil
	}
	features := make([]float64, t.NumFeatures())
	if len(v) == 0 {
		return features
	}
	counts := t.CountVectorizer.Transform(v)

	for i, tf := range counts {
		if tf > 0 && t.DocCount[i] > 0 {
			features[i] = tf * (math.Log(float64(t.NumDocuments)/float64(t.DocCount[i])) + 1)
		}
	}

	return t.Normalizer.Transform(features)
}
