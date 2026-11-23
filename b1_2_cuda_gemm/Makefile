# ==============================================================================
# Makefile para Experimento B1.2: CUDA GEMM & Warp Online Softmax
# Hardware Destino: NVIDIA GeForce GTX 1050 (sm_61, Pascal, 2 GB VRAM)
# ==============================================================================

NVCC := nvcc
ARCH := -arch=$(or $(CUDA_ARCH),sm_61)
CFLAGS := -O3 $(ARCH) -std=c++14 --compiler-options '-fPIC -Wall -Wextra'
LIBS := -lcublas

TARGETS := gemm_bench warp_online_softmax

all: $(TARGETS)

gemm_bench: gemm_bench.cu
	$(NVCC) $(CFLAGS) $< $(LIBS) -o $@

warp_online_softmax: warp_online_softmax.cu
	$(NVCC) $(CFLAGS) $< -o $@

smoke: all
	./gemm_bench --smoke
	./warp_online_softmax --smoke

full: all
	./gemm_bench --full
	./warp_online_softmax --rows 1024 --cols 1024 --warmup 20 --iters 100

clean:
	rm -f $(TARGETS)
