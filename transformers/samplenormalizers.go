package transformers

import "math"

// SampleNormalizerL1 transforms features for single sample to have norm L1=1
type SampleNormalizerL1 struct{}

// Fit is empty, kept only to keep same interface
func (t *SampleNormalizerL1) Fit(_ []float64) {}

// Transform returns L1 normalized vector
func (t *SampleNormalizerL1) Transform(vs []float64) []float64 {
	if vs == nil {
		return nil
	}
	sum := 0.
	for _, v := range vs {
		sum += math.Abs(v)
	}
	vsnorm := make([]float64, len(vs))
	if sum == 0 {
		return vsnorm
	}
	copy(vsnorm, vs)
	for i := range vsnorm {
		vsnorm[i] = vsnorm[i] / sum
	}
	return vsnorm
}

// SampleNormalizerL2 transforms features for single sample to have norm L2=1
type SampleNormalizerL2 struct{}

// Fit is empty, kept only to keep same interface
func (t *SampleNormalizerL2) Fit(_ []float64) {}

// Transform returns L2 normalized vector
func (t *SampleNormalizerL2) Transform(vs []float64) []float64 {
	if vs == nil {
		return nil
	}
	sum := 0.
	for _, v := range vs {
		sum += v * v
	}
	sum = math.Sqrt(sum)
	vsnorm := make([]float64, len(vs))
	if sum == 0 {
		return vsnorm
	}
	copy(vsnorm, vs)
	for i := range vsnorm {
		vsnorm[i] = vsnorm[i] / sum
	}
	return vsnorm
}
