package transformers

import "sort"

// OneHotEncoder encodes string value feature into vector accordingly to order of Values field.
// This less less efficient than map, but more intuitive.
type OneHotEncoder struct {
	Values []string
}

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

func (t *OneHotEncoder) NumFeatures() int {
	return len(t.Values)
}

func (t *OneHotEncoder) Transform(v string) []float64 {
	if len(t.Values) == 0 {
		return nil
	}
	flags := make([]float64, len(t.Values))
	for idx, val := range t.Values {
		if val == v {
			flags[idx] = 1.
		}
	}
	return flags
}

// OrdinalEncoder returns 0 for string that is not found, or else a number for that string
type OrdinalEncoder struct {
	Mapping map[string]float64
}

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

func (t *OrdinalEncoder) Transform(v string) float64 {

	return t.Mapping[v]
}
