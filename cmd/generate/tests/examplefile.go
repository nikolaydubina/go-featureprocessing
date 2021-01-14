package examplemodule

// SomeOther is ignored since there is no gencode command in source file
type SomeOther struct {
	Name1 float64
	Name2 float64
	Name3 string
}

// SomeOtherWithTags is ignored since there is no gencode command in source file, even though it has correct feature tags
type SomeOtherWithTags struct {
	Name1 float64 `feature:"minmax"`
	Name2 float64 `feature:"maxabs"`
	Name3 string  `feature:"onehot"`
	Name4 string  `feature:""`
}

//go:generate go run github.com/nikolaydubina/go-featureprocessing/cmd/generate -struct=AllTransformers

// AllTransformers has all transformer
type AllTransformers struct {
	Name0 int     `feature:"identity"`
	Name1 int32   `feature:"minmax"`
	Name2 float32 `feature:"maxabs"`
	Name3 float64 `feature:"standard"`
	Name4 float64 `feature:"quantile"`
	Name5 string  `feature:"onehot"`
	Name6 string  `feature:"ordinal"`
	Name7 float64 `feature:"kbins"`
	Name8 string  `feature:"countvectorizer"`
	Name9 string  `feature:"tfidf"`
}

//go:generate go run github.com/nikolaydubina/go-featureprocessing/cmd/generate -struct=With32Fields

// With32Fields has many fields
type With32Fields struct {
	Name1  float64 `feature:"minmax"`
	Name2  float64 `feature:"minmax"`
	Name3  float64 `feature:"minmax"`
	Name4  float64 `feature:"minmax"`
	Name5  float64 `feature:"minmax"`
	Name6  float64 `feature:"minmax"`
	Name7  float64 `feature:"minmax"`
	Name8  float64 `feature:"minmax"`
	Name9  float64 `feature:"minmax"`
	Name10 float64 `feature:"minmax"`
	Name11 float64 `feature:"minmax"`
	Name12 float64 `feature:"minmax"`
	Name13 float64 `feature:"minmax"`
	Name14 float64 `feature:"minmax"`
	Name15 float64 `feature:"minmax"`
	Name16 float64 `feature:"minmax"`
	Name17 float64 `feature:"minmax"`
	Name18 float64 `feature:"minmax"`
	Name19 float64 `feature:"minmax"`
	Name21 float64 `feature:"minmax"`
	Name22 float64 `feature:"minmax"`
	Name23 float64 `feature:"minmax"`
	Name24 float64 `feature:"minmax"`
	Name25 float64 `feature:"minmax"`
	Name26 float64 `feature:"minmax"`
	Name27 float64 `feature:"minmax"`
	Name28 float64 `feature:"minmax"`
	Name29 float64 `feature:"minmax"`
	Name30 float64 `feature:"minmax"`
	Name31 float64 `feature:"minmax"`
	Name32 float64 `feature:"minmax"`
}

//go:generate go run github.com/nikolaydubina/go-featureprocessing/cmd/generate -struct=LargeMemoryTransformer

// LargeMemoryTransformer has large memory footprint since each transformer is large
type LargeMemoryTransformer struct {
	Name1 string  `feature:"onehot"`
	Name2 string  `feature:"onehot"`
	Name3 string  `feature:"ordinal"`
	Name4 string  `feature:"ordinal"`
	Name5 float64 `feature:"quantile"`
	Name6 float64 `feature:"quantile"`
	Name7 float64 `feature:"kbins"`
	Name8 float64 `feature:"kbins"`
}

//go:generate go run github.com/nikolaydubina/go-featureprocessing/cmd/generate -struct=WeirdTags

// WeirdTags has unusual but valid tags
type WeirdTags struct {
	OnlyFeature     float64 `feature:"minmax"`
	FeatureNotFirst float64 `json:"name2" feature:"maxabs"`
	FirstFeature    string  `feature:"onehot" json:"some_json_tag"`
	Multiline       float64 `json:"name2" 
	feature:"maxabs"
	`
	WithoutFeatureTag string `json:"with_tag"`
	WeirdTag          string `jsonsadfasd;lk1 asdf;lkads;lfa
	
	asd;flksd;laf
	
	a;sdfkl`
	WithoutTag string

	// UTF-8 is allowed
	A안녕하세요  int    `feature:"minmax"`
	B안녕하세요1 string `feature:"onehot"`
	C안녕하세요0 string `feature:"tfidf"`

	// UTF-8 that starts from non-latin rune is skipped
	안녕하세요2 string `feature:"tfidf"`
}
