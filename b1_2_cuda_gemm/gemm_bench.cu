#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string.h>

/* ==============================================================================
 * MACROS DE DIAGNÓSTICO ESTRICTO Y MANEJO DE ERRORES CUDA / CUBLAS
 * ============================================================================== */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s (código: %d) al ejecutar %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), (int)err, #call); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t stat = (call);                                         \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "[cuBLAS ERROR] %s:%d: estado %d al ejecutar %s\n",\
                    __FILE__, __LINE__, (int)stat, #call);                    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define TILE_DIM 32
#define TILE_DIM_16 16

/* ==============================================================================
 * KERNEL 1: gemm_naive
 * Mapeo: threadIdx.x -> row (fila), threadIdx.y -> col (columna)
 *
 * Impacto arquitectural (Warp Memory Divergence):
 * En CUDA, los hilos de un warp (32 hilos) tienen threadIdx.x consecutivo (0..31).
 * Al mapear threadIdx.x a row, los 32 hilos del warp poseen diferente 'row' y
 * mismo 'col'.
 * - A[row * K + k]: salto de K * 4 bytes entre hilos adyacentes (1 sector L2 de 32 B
 *   por hilo para leer 4 B -> 87.5% de desperdicio de ancho de banda).
 * - B[k * N + col]: todos los hilos leen el mismo dato -> broadcast de warp.
 * - C[row * N + col]: escritura no coalescente a saltos de N * 4 bytes.
 * ============================================================================== */
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = sum;
    }
}

/* ==============================================================================
 * KERNEL 2: gemm_coalesced
 * Mapeo: threadIdx.x -> col (columna), threadIdx.y -> row (fila)
 *
 * Impacto arquitectural (Memory Coalescing & Broadcast):
 * En este caso, los hilos adyacentes de un warp (threadIdx.x = 0..31) comparten 'row'
 * y varían en 'col' de forma consecutiva (col, col+1, ...).
 * - Lectura de A[row * K + k]: 'row' y 'k' son IDÉNTICOS para los 32 hilos.
 *   El SM realiza un único broadcast desde L1/L2 cache a todo el warp (1 ciclo).
 * - Lectura de B[k * N + col]: Las direcciones son contiguas (k*N + col, k*N + col+1...).
 *   Los 32 accesos de 4 bytes se agrupan en una línea L2 de 128 B (4 sectores de 32 B).
 * - Escritura de C[row * N + col]: Perfectamente coalescente (128 bytes contiguos).
 * ============================================================================== */
__global__ void gemm_coalesced(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = sum;
    }
}

/* ==============================================================================
 * KERNEL 3: gemm_smem_tiled_16
 * Tiling cooperativo en Shared Memory de 16x16 floats (256 hilos/bloque)
 *
 * Compromiso arquitectural frente a 32x32 en Pascal (sm_61):
 * - Tile: 16x16 floats (As: 1 KB, Bs: 1 KB -> 2 KB SMEM por bloque).
 * - Ocupación más fina: 256 hilos (8 warps) permiten hasta 8 bloques/SM (2048 hilos)
 *   y evitan caer de golpe al 50% si el kernel supera 32 registros/hilo; además cada
 *   __syncthreads() solo detiene 8 warps y permite a otros bloques ocultarlo.
 * - Coste: factor de reutilización 16x (la mitad que 32x32) y doble de barreras.
 * - Cero bank conflicts: en warp de 32 hilos con blockDim(16,16), los medios warps
 *   ty=0 y ty=1 acceden a filas distintas de As[ty][k] (16 apart) y hacen broadcast
 *   por medio warp en Bs[k][tx].
 * ============================================================================== */
__global__ void gemm_smem_tiled_16(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[TILE_DIM_16][TILE_DIM_16];
    __shared__ float Bs[TILE_DIM_16][TILE_DIM_16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_DIM_16 + tx;
    int row = blockIdx.y * TILE_DIM_16 + ty;

    float sum = 0.0f;
    int numTiles = (K + TILE_DIM_16 - 1) / TILE_DIM_16;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM_16 + tx;
        int b_row = t * TILE_DIM_16 + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM_16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] = sum;
    }
}

/* ==============================================================================
 * KERNEL 4: gemm_smem_tiled
 * Tiling cooperativo en Shared Memory de 32x32 floats con doble __syncthreads()
 *
 * Impacto arquitectural (Data Reuse & Bank Conflict Free):
 * - Tile: 32x32 floats (As: 4 KB, Bs: 4 KB -> Total 8 KB SMEM por bloque).
 *   En Pascal GP107 (GTX 1050), cada bloque de 1024 hilos ocupa 32 warps (máx. 2
 *   bloques/SM -> cae al 50% de ocupación si supera 32 registros/hilo).
 * - Factor de Reutilización: Cada dato transferido desde DRAM GDDR5 a SMEM es reutilizado
 *   32 veces por los hilos del bloque, reduciendo el tráfico global DRAM en hasta 32x.
 * - Shared Memory Banks:
 *   - Lectura As[ty][k]: ty constante por warp, k constante en cada iteración del bucle.
 *     Acceso broadcast de SMEM (0 bank conflicts).
 *   - Lectura Bs[k][tx]: tx varía de 0 a 31. Cada hilo i del warp accede al banco i.
 *     Acceso 100% libre de bank conflicts.
 * - Doble barrera __syncthreads():
 *   1) Primera barrera: garantiza que todo el tile cooperativo esté en SMEM antes de calcular.
 *   2) Segunda barrera: garantiza que todos los hilos terminen de consumir el tile antes
 *      de que el hilo cooperativo sobreescriba As y Bs en la siguiente iteración.
 * ============================================================================== */
__global__ void gemm_smem_tiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    float sum = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // Carga colaborativa con verificación de bordes
        As[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0f;

        // Barrera 1: Tile completamente cargado en SMEM
        __syncthreads();

        // Multiplicación en registros y SMEM
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Barrera 2: Cálculo completado antes de actualizar SMEM
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] = sum;
    }
}

/* ==============================================================================
 * EJECUCIÓN DE CUBLAS SGEMM DE REFERENCIA
 * Mapeo algebraico Row-Major a Column-Major:
 * C_rm = A_rm * B_rm
 * Dado que un tensor row-major (M x K) en memoria es idéntico a un col-major (K x M)^T,
 * la relación transpuesta es:
 * C_cm = B_cm * A_cm
 * Donde B_cm es (N x K) con ldb = N, A_cm es (K x M) con lda = K, y C_cm es (N x M) con ldc = N.
 * Esto ejecuta exactamente C = A * B en memoria row-major sin copias ni transposiciones.
 * ============================================================================== */
void run_cublas_gemm(cublasHandle_t handle,
                     const float* d_A, const float* d_B, float* d_C,
                     int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    ));
}

/* ==============================================================================
 * VERIFICACIÓN NUMÉRICA CONTRA CUBLAS
 * Tolerancia: max_abs_diff < 1e-3 o max_rel_diff < 1e-3
 * ============================================================================== */
bool verify_results(const float* h_cand, const float* h_ref, int size,
                    float* out_max_abs, float* out_max_rel) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool pass = true;
    const float tol = 1e-3f;

    for (int i = 0; i < size; ++i) {
        float cand = h_cand[i];
        float ref = h_ref[i];
        float diff = fabsf(cand - ref);
        float denom = fmaxf(fabsf(ref), 1e-5f);
        float rel_diff = diff / denom;

        if (diff > max_abs) max_abs = diff;
        if (rel_diff > max_rel) max_rel = rel_diff;

        if (diff > tol && rel_diff > tol) {
            pass = false;
        }
    }

    if (out_max_abs) *out_max_abs = max_abs;
    if (out_max_rel) *out_max_rel = max_rel;
    return pass;
}

/* ==============================================================================
 * ESTRUCTURA DE RESULTADOS DEL BENCHMARK
 * ============================================================================== */
typedef struct {
    char name[32];
    int M, N, K;
    float avg_ms;
    double gflops;
    double speedup;
    bool verified;
    float max_abs_err;
    float max_rel_err;
} GemmResult;

/* ==============================================================================
 * HELPER DE EJECUCIÓN Y VERIFICACIÓN PARA KERNELS GEMM PERSONALIZADOS
 * ============================================================================== */
template <typename LaunchFn>
void benchmark_custom_kernel(const char* name, LaunchFn launch_fn,
                             dim3 grid, dim3 block,
                             const float* d_A, const float* d_B, float* d_C_cand,
                             float* h_C_cand, const float* h_C_ref,
                             int M, int N, int K, size_t bytes_C,
                             int warmup_iters, int bench_iters,
                             double total_flops, cudaEvent_t ev_start, cudaEvent_t ev_stop,
                             GemmResult* out_res) {
    CUDA_CHECK(cudaMemset(d_C_cand, 0, bytes_C));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(grid, block, d_A, d_B, d_C_cand, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int i = 0; i < bench_iters; ++i) {
        launch_fn(grid, block, d_A, d_B, d_C_cand, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    float avg_ms = ms / (float)bench_iters;
    double gflops = total_flops / (avg_ms * 1e6);

    CUDA_CHECK(cudaMemcpy(h_C_cand, d_C_cand, bytes_C, cudaMemcpyDeviceToHost));
    float abs_err = 0.0f, rel_err = 0.0f;
    bool ok = verify_results(h_C_cand, h_C_ref, M * N, &abs_err, &rel_err);

    snprintf(out_res->name, sizeof(out_res->name), "%s", name);
    out_res->M = M; out_res->N = N; out_res->K = K;
    out_res->avg_ms = avg_ms;
    out_res->gflops = gflops;
    out_res->speedup = 1.0;
    out_res->verified = ok;
    out_res->max_abs_err = abs_err;
    out_res->max_rel_err = rel_err;
}

static void launch_naive(dim3 g, dim3 b, const float* A, const float* B, float* C, int M, int N, int K) {
    gemm_naive<<<g, b>>>(A, B, C, M, N, K);
}
static void launch_coalesced(dim3 g, dim3 b, const float* A, const float* B, float* C, int M, int N, int K) {
    gemm_coalesced<<<g, b>>>(A, B, C, M, N, K);
}
static void launch_smem_16(dim3 g, dim3 b, const float* A, const float* B, float* C, int M, int N, int K) {
    gemm_smem_tiled_16<<<g, b>>>(A, B, C, M, N, K);
}
static void launch_smem_32(dim3 g, dim3 b, const float* A, const float* B, float* C, int M, int N, int K) {
    gemm_smem_tiled<<<g, b>>>(A, B, C, M, N, K);
}

/* ==============================================================================
 * FUNCIÓN PRINCIPAL DE EJECUCIÓN POR TAMAÑO
 * ============================================================================== */
void benchmark_dimension(int M, int N, int K, int warmup_iters, int bench_iters,
                         cublasHandle_t handle, GemmResult* out_results) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);
    size_t total_vram = bytes_A + bytes_B + 2 * bytes_C; // d_A, d_B, d_C_cand, d_C_ref

    size_t free_vram = 0, total_gpu_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_gpu_vram));
    if (total_vram > (size_t)((double)free_vram * 0.90)) {
        fprintf(stderr,
                "[FATAL ERROR] Dimensión %dx%dx%d requiere %.2f MB VRAM y excede el 90%% de VRAM libre (%.2f MB).\n",
                M, N, K, (double)total_vram / (1024.0 * 1024.0),
                (double)free_vram / (1024.0 * 1024.0));
        exit(EXIT_FAILURE);
    }

    printf("\n>>> Evaluando GEMM FP32: M=%d, N=%d, K=%d (VRAM total búferes: %.2f MB)\n",
           M, N, K, (double)total_vram / (1024.0 * 1024.0));

    // Alojamiento en Host
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C_cand = (float*)malloc(bytes_C);
    float* h_C_ref = (float*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C_cand || !h_C_ref) {
        fprintf(stderr, "[FATAL ERROR] Fallo al alojar memoria en Host.\n");
        exit(EXIT_FAILURE);
    }

    // Inicialización determinista pseudoaleatoria en [-1.0, 1.0]
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    // Alojamiento en Device (VRAM)
    float *d_A, *d_B, *d_C_cand, *d_C_ref;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C_cand, bytes_C));
    CUDA_CHECK(cudaMalloc((void**)&d_C_ref, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Eventos CUDA de cronometraje
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_flops = 2.0 * (double)M * (double)N * (double)K;

    // --------------------------------------------------------------------------
    // 0. cuBLAS (Referencia dorada)
    // --------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        run_cublas_gemm(handle, d_A, d_B, d_C_ref, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int i = 0; i < bench_iters; ++i) {
        run_cublas_gemm(handle, d_A, d_B, d_C_ref, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_cublas = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, ev_start, ev_stop));
    float avg_ms_cublas = ms_cublas / (float)bench_iters;
    double gflops_cublas = total_flops / (avg_ms_cublas * 1e6);

    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    snprintf(out_results[0].name, sizeof(out_results[0].name), "cublasSgemm (Ref)");
    out_results[0].M = M; out_results[0].N = N; out_results[0].K = K;
    out_results[0].avg_ms = avg_ms_cublas;
    out_results[0].gflops = gflops_cublas;
    out_results[0].speedup = 1.0;
    out_results[0].verified = true;
    out_results[0].max_abs_err = 0.0f;
    out_results[0].max_rel_err = 0.0f;

    // --------------------------------------------------------------------------
    // 1. gemm_naive (tx -> row, ty -> col)
    // 2. gemm_coalesced (tx -> col, ty -> row)
    // 3. gemm_smem_tiled_16 (tile 16x16, 256 threads/block)
    // 4. gemm_smem_tiled (tile 32x32, 1024 threads/block)
    // --------------------------------------------------------------------------
    dim3 block32(TILE_DIM, TILE_DIM);
    dim3 grid_naive((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 grid_32((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block16(TILE_DIM_16, TILE_DIM_16);
    dim3 grid_16((N + TILE_DIM_16 - 1) / TILE_DIM_16, (M + TILE_DIM_16 - 1) / TILE_DIM_16);

    benchmark_custom_kernel("gemm_naive", launch_naive, grid_naive, block32,
                            d_A, d_B, d_C_cand, h_C_cand, h_C_ref,
                            M, N, K, bytes_C, warmup_iters, bench_iters,
                            total_flops, ev_start, ev_stop, &out_results[1]);

    benchmark_custom_kernel("gemm_coalesced", launch_coalesced, grid_32, block32,
                            d_A, d_B, d_C_cand, h_C_cand, h_C_ref,
                            M, N, K, bytes_C, warmup_iters, bench_iters,
                            total_flops, ev_start, ev_stop, &out_results[2]);

    benchmark_custom_kernel("gemm_smem_tiled_16", launch_smem_16, grid_16, block16,
                            d_A, d_B, d_C_cand, h_C_cand, h_C_ref,
                            M, N, K, bytes_C, warmup_iters, bench_iters,
                            total_flops, ev_start, ev_stop, &out_results[3]);

    benchmark_custom_kernel("gemm_smem_tiled", launch_smem_32, grid_32, block32,
                            d_A, d_B, d_C_cand, h_C_cand, h_C_ref,
                            M, N, K, bytes_C, warmup_iters, bench_iters,
                            total_flops, ev_start, ev_stop, &out_results[4]);

    double base_ms = (double)out_results[1].avg_ms;
    out_results[0].speedup = base_ms / (double)out_results[0].avg_ms;
    out_results[1].speedup = 1.0;
    out_results[2].speedup = base_ms / (double)out_results[2].avg_ms;
    out_results[3].speedup = base_ms / (double)out_results[3].avg_ms;
    out_results[4].speedup = base_ms / (double)out_results[4].avg_ms;

    // Liberación de recursos
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cand));
    CUDA_CHECK(cudaFree(d_C_ref));
    free(h_A);
    free(h_B);
    free(h_C_cand);
    free(h_C_ref);
}

/* ==============================================================================
 * IMPRESIÓN Y SERIALIZACIÓN
 * ============================================================================== */
void print_markdown_table(const GemmResult* res, int count) {
    printf("\n| Kernel GEMM | Dimensión (MxNxK) | Latencia (ms) | Rendimiento (GFLOP/s) | Speedup vs Naive | Verificación (Tol 1e-3) |\n");
    printf("| :--- | :---: | :---: | :---: | :---: | :---: |\n");
    for (int i = 0; i < count; ++i) {
        printf("| **%-18s** | %4dx%4dx%4d | %10.3f | %18.2f | %13.2fx | %s (err: %.2e) |\n",
               res[i].name, res[i].M, res[i].N, res[i].K,
               res[i].avg_ms, res[i].gflops, res[i].speedup,
               res[i].verified ? "PASS [OK]" : "FAIL [ERROR]",
               res[i].max_abs_err);
    }
}

int main(int argc, char** argv) {
    bool smoke_mode = false;
    bool full_mode = false;
    int custom_dim = 0;
    int warmup_iters = 10;
    int bench_iters = 50;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--smoke") == 0 || strcmp(argv[i], "--smoke-test") == 0) {
            smoke_mode = true;
        } else if (strcmp(argv[i], "--full") == 0) {
            full_mode = true;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            custom_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        }
    }

    if (warmup_iters < 0 || bench_iters <= 0 || custom_dim < 0) {
        fprintf(stderr, "[ERROR] Parámetros inválidos (warmup >= 0, iters > 0, size > 0).\n");
        return EXIT_FAILURE;
    }

    if (!smoke_mode && !full_mode && custom_dim == 0) {
        smoke_mode = true; // Por defecto modo smoke seguro
    } else if (smoke_mode && full_mode) {
        fprintf(stderr, "[WARN] Se especificaron --smoke y --full a la vez; priorizando --full.\n");
        smoke_mode = false;
    }

    // Inicializar dispositivo CUDA
    int dev_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    CUDA_CHECK(cudaSetDevice(dev_id));

    printf("==============================================================================\n");
    printf("  BENCHMARK GEMM FP32: NVIDIA %s (sm_%d%d, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Configuración: Warmup=%d iters, Medición=%d iters\n", warmup_iters, bench_iters);
    printf("==============================================================================\n");

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    int dims[4];
    int num_dims = 0;

    if (custom_dim > 0) {
        dims[num_dims++] = custom_dim;
    } else if (full_mode) {
        dims[num_dims++] = 1024;
        dims[num_dims++] = 2048;
    } else if (smoke_mode) {
        dims[num_dims++] = 256;
    }

    GemmResult all_results[20];
    int total_runs = 0;

    for (int d = 0; d < num_dims; ++d) {
        int dim = dims[d];
        benchmark_dimension(dim, dim, dim, warmup_iters, bench_iters, cublas_handle, &all_results[total_runs]);
        total_runs += 5;
    }

    print_markdown_table(all_results, total_runs);

    // Verificación global de salida
    bool all_ok = true;
    for (int i = 0; i < total_runs; ++i) {
        if (!all_results[i].verified) all_ok = false;
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    if (!all_ok) {
        fprintf(stderr, "\n[FATAL ERROR] Al menos un kernel fallo la verificacion numerica contra cuBLAS!\n");
        return EXIT_FAILURE;
    }

    printf("\n[SUCCESS] Todos los kernels GEMM pasaron la verificacion numerica con exito.\n");
    return EXIT_SUCCESS;
}
