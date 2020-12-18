package structtransformer_test

import (
	"math/rand"
	"testing"

	. "github.com/nikolaydubina/go-featureprocessing/structtransformer"
	. "github.com/nikolaydubina/go-featureprocessing/transformers"

	"github.com/stretchr/testify/assert"
)

func TestStructTransformer(t *testing.T) {
	t.Run("test transform basic", func(t *testing.T) {
		type S struct {
			Age    int     `feature:"minmax"`
			Salary float64 `feature:"standard"`
			Gender string  `feature:"onehot"`
			City   string  `feature:"ordinal"`
		}

		tr := StructTransformer{Transformers: []interface{}{
			&MinMaxScaler{Min: 1, Max: 10},
			&StandardScaler{Mean: 15, STD: 2.5},
			&OneHotEncoder{Values: []string{"male", "female"}},
			&OrdinalEncoder{Mapping: map[string]float64{"city-A": 1, "city-B": 2}},
		}}

		assert.Equal(t, []float64{1, 1, 0, 1, 2}, tr.Transform(S{Age: 23, Salary: 17.5, Gender: "female", City: "city-B"}))
	})

	t.Run("test transform struct has fields but transformers missing", func(t *testing.T) {
		type S struct {
			Age    int     `feature:"minmax"`
			Salary float64 `feature:"standard"`
			Gender string  `feature:"onehot"`
			City   string  `feature:"ordinal"`
		}

		tr := StructTransformer{}
		assert.Equal(t, []float64(nil), tr.Transform(S{Age: 23, Salary: 17.5, Gender: "female", City: "city-B"}))
	})

	t.Run("test transform nil", func(t *testing.T) {
		tr := StructTransformer{}
		assert.Equal(t, []float64(nil), tr.Transform(nil))
	})

	t.Run("test transform nil pointer to struct", func(t *testing.T) {
		type S struct {
			Age    int     `feature:"minmax"`
			Salary float64 `feature:"standard"`
			Gender string  `feature:"onehot"`
			City   string  `feature:"ordinal"`
		}
		var s S
		tr := StructTransformer{}
		assert.Equal(t, []float64(nil), tr.Transform(&s))
	})

	t.Run("test transform nil interface", func(t *testing.T) {
		type S interface {
			Get() int
		}
		var s S
		tr := StructTransformer{}
		assert.Equal(t, []float64(nil), tr.Transform(&s))
		assert.Equal(t, []float64(nil), tr.Transform(s))
	})
}

func BenchmarkStructTransformer_Transform_Small(b *testing.B) {
	type S struct {
		Age    int     `feature:"minmax"`
		Salary float64 `feature:"standard"`
		Gender string  `feature:"onehot"`
		City   string  `feature:"ordinal"`
	}

	tr := StructTransformer{Transformers: []interface{}{
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&OneHotEncoder{Values: []string{"male", "female"}},
		&OrdinalEncoder{Mapping: map[string]float64{"city-A": 1, "city-B": 2}},
	}}

	s := S{
		Age:    23,
		Salary: 17.5,
		Gender: "female",
		City:   "city-B",
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		tr.Transform(s)
	}
}

func randomInt(min, max int) int {
	return min + rand.Intn(max-min)
}

func randomString(len int) string {
	bytes := make([]byte, len)
	for i := 0; i < len; i++ {
		bytes[i] = byte(randomInt(65, 90))
	}
	return string(bytes)
}

func randomSliceFloat64(num int) []float64 {
	ret := make([]float64, num)
	for i := 0; i < num; i++ {
		ret[i] = rand.Float64()
	}
	return ret
}

func randomSliceString(num int, strlen int) []string {
	ret := make([]string, num)
	for i := 0; i < num; i++ {
		ret[i] = randomString(strlen)
	}
	return ret
}

func randomMappingString(num int, strlen int) map[string]float64 {
	ret := make(map[string]float64)
	for i := 0; i < num; i++ {
		ret[randomString(strlen)] = float64(i)
	}
	return ret
}

func getAnyKeyFromMap(mp map[string]float64) string {
	for k, _ := range mp {
		return k
	}
	return ""
}

func benchLargeTransformer(b *testing.B, numelem int) {
	type S struct {
		Name1 string  `feature:"onehot"`
		Name2 string  `feature:"onehot"`
		Name3 string  `feature:"ordinal"`
		Name4 string  `feature:"ordinal"`
		Name5 float64 `feature:"quantile"`
		Name6 float64 `feature:"quantile"`
		Name7 float64 `feature:"kbins"`
		Name8 float64 `feature:"kbins"`
	}

	tr := StructTransformer{Transformers: []interface{}{
		&OneHotEncoder{Values: randomSliceString(numelem, 20)},
		&OneHotEncoder{Values: randomSliceString(numelem, 20)},
		&OrdinalEncoder{Mapping: randomMappingString(numelem, 20)},
		&OrdinalEncoder{Mapping: randomMappingString(numelem, 20)},
		&QuantileScaler{Quantiles: randomSliceFloat64(numelem), NQuantiles: numelem},
		&QuantileScaler{Quantiles: randomSliceFloat64(numelem), NQuantiles: numelem},
		&KBinsDiscretizer{QuantileScaler{Quantiles: randomSliceFloat64(numelem), NQuantiles: numelem}},
		&KBinsDiscretizer{QuantileScaler{Quantiles: randomSliceFloat64(numelem), NQuantiles: numelem}},
	}}

	s := S{
		Name1: tr.Transformers[0].(*OneHotEncoder).Values[randomInt(1, numelem-1)],
		Name2: tr.Transformers[1].(*OneHotEncoder).Values[randomInt(1, numelem-1)],
		Name3: getAnyKeyFromMap(tr.Transformers[2].(*OrdinalEncoder).Mapping),
		Name4: getAnyKeyFromMap(tr.Transformers[3].(*OrdinalEncoder).Mapping),
		Name5: tr.Transformers[4].(*QuantileScaler).Quantiles[randomInt(1, numelem-1)],
		Name6: tr.Transformers[5].(*QuantileScaler).Quantiles[randomInt(1, numelem-1)],
		Name7: tr.Transformers[6].(*KBinsDiscretizer).Quantiles[randomInt(1, numelem-1)],
		Name8: tr.Transformers[7].(*KBinsDiscretizer).Quantiles[randomInt(1, numelem-1)],
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		tr.Transform(s)
	}
}

func BenchmarkStructTransformer_Transform_LargeComposites_100elements(b *testing.B) {
	benchLargeTransformer(b, 100)
}

func BenchmarkStructTransformer_Transform_LargeComposites_1000elements(b *testing.B) {
	benchLargeTransformer(b, 1000)
}

func BenchmarkStructTransformer_Transform_LargeComposites_10000elements(b *testing.B) {
	benchLargeTransformer(b, 10000)
}

func BenchmarkStructTransformer_Transform_LargeComposites_100000elements(b *testing.B) {
	benchLargeTransformer(b, 100000)
}

func BenchmarkStructTransformer_Transform_32fields(b *testing.B) {

	type S struct {
		F1  float64 `feature:"minmax"`
		F2  float64 `feature:"standard"`
		F3  float64 `feature:"minmax"`
		F4  float64 `feature:"standard"`
		F5  float64 `feature:"minmax"`
		F6  float64 `feature:"standard"`
		F7  float64 `feature:"minmax"`
		F8  float64 `feature:"standard"`
		F9  float64 `feature:"minmax"`
		F10 float64 `feature:"standard"`
		F11 float64 `feature:"minmax"`
		F12 float64 `feature:"standard"`
		F13 float64 `feature:"minmax"`
		F14 float64 `feature:"standard"`
		F15 float64 `feature:"minmax"`
		F16 float64 `feature:"standard"`
		F17 float64 `feature:"minmax"`
		F18 float64 `feature:"standard"`
		F19 float64 `feature:"minmax"`
		F20 float64 `feature:"standard"`
		F21 float64 `feature:"minmax"`
		F22 float64 `feature:"standard"`
		F23 float64 `feature:"minmax"`
		F24 float64 `feature:"standard"`
		F25 float64 `feature:"minmax"`
		F26 float64 `feature:"standard"`
		F27 float64 `feature:"minmax"`
		F28 float64 `feature:"standard"`
		F29 float64 `feature:"minmax"`
		F30 float64 `feature:"standard"`
		F31 float64 `feature:"minmax"`
		F32 float64 `feature:"standard"`
	}

	tr := StructTransformer{Transformers: []interface{}{
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
		&MinMaxScaler{Min: 1, Max: 10},
		&StandardScaler{Mean: 15, STD: 2.5},
	}}

	s := S{
		F1:  1231231.123,
		F2:  1231231.123,
		F3:  1231231.123,
		F4:  1231231.123,
		F5:  1231231.123,
		F6:  1231231.123,
		F7:  1231231.123,
		F8:  1231231.123,
		F9:  1231231.123,
		F10: 1231231.123,
		F11: 1231231.123,
		F12: 1231231.123,
		F13: 1231231.123,
		F14: 1231231.123,
		F15: 1231231.123,
		F16: 1231231.123,
		F17: 1231231.123,
		F18: 1231231.123,
		F19: 1231231.123,
		F20: 1231231.123,
		F21: 1231231.123,
		F22: 1231231.123,
		F23: 1231231.123,
		F24: 1231231.123,
		F25: 1231231.123,
		F26: 1231231.123,
		F27: 1231231.123,
		F28: 1231231.123,
		F29: 1231231.123,
		F30: 1231231.123,
		F31: 1231231.123,
		F32: 1231231.123,
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		tr.Transform(s)
	}
}
