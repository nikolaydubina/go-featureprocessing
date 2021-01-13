package transformers

import "sort"

// OneHotEncoder encodes string value feature into vector accordingly to order of Values field.
// This less less efficient than map, but more intuitive.
type OneHotEncoder struct {
	Values []string
}

// Fit assigns each value from inputs a number, then makes string of values and sorts values
func (t *OneHotEncoder) Fit(vals []string) {
	if vals == nil {
		return
	}
	mp := make(map[string]int)
	for _, v := range vals {
		if _, ok := mp[v]; !ok {
			mp[v] = len(mp)
		}
	}
	t.Values = make([]string, len(mp))
	for val, idx := range mp {
		t.Values[idx] = val
	}
	sort.Slice(t.Values, func(i, j int) bool {
		return mp[t.Values[i]] < mp[t.Values[j]]
	})
}

// NumFeatures returns number of features one field is expanded
func (t *OneHotEncoder) NumFeatures() int {
	return len(t.Values)
}

// Transform assigns 1 to value that is found
func (t *OneHotEncoder) Transform(v string) []float64 {
	if t == nil || len(t.Values) == 0 {
		return nil
	}
	flags := make([]float64, len(t.Values))
	t.TransformInplace(flags, v)
	return flags
}

// TransformInplace assigns 1 to value that is found, inplace
func (t *OneHotEncoder) TransformInplace(dest []float64, v string) {
	if t == nil || len(t.Values) == 0 || len(dest) != len(t.Values) {
		return
	}
	for idx, val := range t.Values {
		if val == v {
			dest[idx] = 1
		} else {
			dest[idx] = 0
		}
	}
}

// FeatureNames returns names of each produced value.
func (t *OneHotEncoder) FeatureNames() []string {
	if t == nil || len(t.Values) == 0 {
		return nil
	}
	names := make([]string, t.NumFeatures())
	for i, w := range t.Values {
		names[i] = w
	}
	return names
}

// OrdinalEncoder returns 0 for string that is not found, or else a number for that string
type OrdinalEncoder struct {
	Mapping map[string]float64
}

// Fit assigns each word value from 1 to N
func (t *OrdinalEncoder) Fit(vals []string) {
	if len(vals) == 0 {
		return
	}
	t.Mapping = make(map[string]float64)
	for _, v := range vals {
		if _, ok := t.Mapping[v]; !ok {
			t.Mapping[v] = float64(len(t.Mapping) + 1)
		}
	}
}

// Transform returns number of input, if not found returns zero value which is 0
func (t *OrdinalEncoder) Transform(v string) float64 {
	return t.Mapping[v]
}
