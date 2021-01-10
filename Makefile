test:
	go generate ./...
	go test -covermode=atomic ./...

bench:
	go test -bench=. -benchtime=3s -benchmem ./...

bench-transform:
	go test -run ".*Transform" -bench=. -benchtime=3s -benchmem ./...

profile:
	$(shell mkdir -p benchmark_profiles)
	go test -bench=BenchmarkWith32FieldsFeatureTransformer_Transform -benchtime=3s -benchmem -memprofile benchmark_profiles/codegen_transform_mem.profile -cpuprofile benchmark_profiles/codegen_transform_cpu.profile ./cmd/generate/tests
	go test -bench=BenchmarkStructTransformer_Transform_32fields -benchtime=3s -benchmem -memprofile benchmark_profiles/reflect_transform_mem.profile -cpuprofile benchmark_profiles/reflect_transform_cpu.profile ./structtransformer
