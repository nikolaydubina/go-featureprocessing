test:
	go generate ./...
	go test -covermode=atomic ./...

bench:
	GOMAXPROCS=8 go test -timeout=1h -bench=. -benchtime=10s -benchmem ./...

bench-transform:
	GOMAXPROCS=8 go test -bench=_Transform -benchtime=10s -benchmem ./...

bench-transform-employee:
	GOMAXPROCS=8 go test -bench=BenchmarkEmployeeFeatureTransformer_Transform -benchtime=10s -benchmem ./...

profile:
	$(shell mkdir -p docs/benchmark_profiles)
	go test -bench=BenchmarkWith32FieldsFeatureTransformer_Transform -benchtime=3s -benchmem -memprofile docs/benchmark_profiles/codegen_transform_mem.profile -cpuprofile docs/benchmark_profiles/codegen_transform_cpu.profile ./cmd/generate/tests
	go test -bench=BenchmarkStructTransformer_Transform_32fields -benchtime=3s -benchmem -memprofile docs/benchmark_profiles/reflect_transform_mem.profile -cpuprofile docs/benchmark_profiles/reflect_transform_cpu.profile ./structtransformer
