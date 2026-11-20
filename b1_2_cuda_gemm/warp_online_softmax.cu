#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string.h>

/* ==============================================================================
 * MACROS DE DIAGNÓSTICO ESTRICTO CUDA
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

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

/* ==============================================================================
 * OPERADOR ASOCIATIVO DE REDUCCIÓN INTRA-WARP (SHUFFLE)
 * Combina dos tuplas (m1, d1) y (m2, d2):
 * m_new = max(m1, m2)
 * d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new)
 * Sin usar Shared Memory (__shared__), 100% registros vía __shfl_down_sync.
 * ============================================================================== */
__device__ __forceinline__ void warp_reduce_online_combine(float &m, float &d) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(FULL_MASK, m, offset);
        float other_d = __shfl_down_sync(FULL_MASK, d, offset);

        float m_max = fmaxf(m, other_m);
        // Reescalado de suma de exponenciales al nuevo máximo común
        d = d * __expf(m - m_max) + other_d * __expf(other_m - m_max);
        m = m_max;
    }
}

/* ==============================================================================
 * REDUCCIÓN INTRA-WARP BÁSICA (MÁXIMO Y SUMA SEPARADOS PARA 3-PASS)
 * ============================================================================== */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

/* ==============================================================================
 * KERNEL 1: online_softmax_warp_kernel (1 Pasada de Reducción)
 * - 1 Warp por fila.
 * - Fase 1: Recorrido coalescente en 1 pasada calculando (m, d) acumulativo.
 * - Reducción intra-warp con warp_reduce_online_combine en registros.
 * - Broadcast de (m, d) a todos los 32 hilos del warp vía __shfl_sync.
 * - Fase 2: Escritura directa y normalizada del resultado en memoria global.
 * - Cero uso de Shared Memory. Acceso coalescente perfecto a DRAM.
 * ============================================================================== */
__global__ void online_softmax_warp_kernel(const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int M, int N) {
    // 1 warp maneja 1 fila
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int row = warp_id_global;
    if (row >= M) return;

    const float* row_in = X + row * N;
    float* row_out = Y + row * N;

    // Estado online local al hilo: (m: máximo actual, d: denominador actual)
    float m = -INFINITY;
    float d = 0.0f;

    // Recorrido de la fila con stride de warp (coalescencia de 128 bytes por iteración)
    for (int col = lane; col < N; col += WARP_SIZE) {
        float x = row_in[col];
        float m_new = fmaxf(m, x);
        d = d * __expf(m - m_new) + __expf(x - m_new);
        m = m_new;
    }

    // Reducción intra-warp de (m, d) en árbol mariposa (5 ciclos de shuffle)
    warp_reduce_online_combine(m, d);

    // Broadcast del resultado de lane 0 a todos los hilos del warp
    m = __shfl_sync(FULL_MASK, m, 0);
    d = __shfl_sync(FULL_MASK, d, 0);

    float inv_d = 1.0f / d;

    // Escritura de la salida normalizada (acceso coalescente)
    for (int col = lane; col < N; col += WARP_SIZE) {
        row_out[col] = __expf(row_in[col] - m) * inv_d;
    }
}

/* ==============================================================================
 * KERNEL 2: standard_3pass_softmax_warp_kernel (Referencia GPU 3 Pasadas)
 * - Pasada 1: Lee X para encontrar max_val.
 * - Pasada 2: Lee X para acumular sum(exp(x - max_val)).
 * - Pasada 3: Lee X, calcula exp(x - max_val) / sum y escribe Y.
 * ============================================================================== */
__global__ void standard_3pass_softmax_warp_kernel(const float* __restrict__ X,
                                                  float* __restrict__ Y,
                                                  int M, int N) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int row = warp_id_global;
    if (row >= M) return;

    const float* row_in = X + row * N;
    float* row_out = Y + row * N;

    // Pasada 1: Máximo de la fila
    float local_max = -INFINITY;
    for (int col = lane; col < N; col += WARP_SIZE) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(FULL_MASK, row_max, 0);

    // Pasada 2: Suma de exponenciales
    float local_sum = 0.0f;
    for (int col = lane; col < N; col += WARP_SIZE) {
        local_sum += __expf(row_in[col] - row_max);
    }
    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(FULL_MASK, row_sum, 0);
    float inv_sum = 1.0f / row_sum;

    // Pasada 3: Normalización y escritura
    for (int col = lane; col < N; col += WARP_SIZE) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

/* ==============================================================================
 * REFERENCIA DE CPU (DOUBLE PRECISION)
 * ============================================================================== */
void cpu_softmax_reference(const float* X, float* Y, int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* row_in = X + r * N;
        float* row_out = Y + r * N;

        // Pasada 1: Max
        double max_val = -1e30;
        for (int c = 0; c < N; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        // Pasada 2: Denominador
        double sum = 0.0;
        for (int c = 0; c < N; ++c) {
            sum += exp((double)row_in[c] - max_val);
        }
        double inv_sum = 1.0 / sum;

        // Pasada 3: Salida
        for (int c = 0; c < N; ++c) {
            row_out[c] = (float)(exp((double)row_in[c] - max_val) * inv_sum);
        }
    }
}

/* ==============================================================================
 * VERIFICACIÓN NUMÉRICA ESTRICTA (Tolerancia 1e-5)
 * ============================================================================== */
bool verify_softmax(const float* candidate, const float* reference, int total_elements,
                    float* out_max_abs, float* out_max_rel) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool pass = true;
    const float tol = 1e-5f;

    for (int i = 0; i < total_elements; ++i) {
        float cand = candidate[i];
        float ref = reference[i];
        float diff = fabsf(cand - ref);
        float rel = diff / fmaxf(fabsf(ref), 1e-7f);

        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;

        if (diff > tol) {
            pass = false;
        }
    }

    if (out_max_abs) *out_max_abs = max_abs;
    if (out_max_rel) *out_max_rel = max_rel;
    return pass;
}

/* ==============================================================================
 * MAIN
 * ============================================================================== */
int main(int argc, char** argv) {
    int M = 1024; // Filas
    int N = 1024; // Columnas
    int warmup_iters = 20;
    int bench_iters = 100;
    bool smoke_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--smoke") == 0 || strcmp(argv[i], "--smoke-test") == 0) {
            smoke_mode = true;
            warmup_iters = 5;
            bench_iters = 20;
        } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        }
    }

    int dev_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    CUDA_CHECK(cudaSetDevice(dev_id));

    printf("==============================================================================\n");
    printf("  BENCHMARK WARP ONLINE SOFTMAX: NVIDIA %s (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("  Matriz: %d x %d (%.2f MB FP32), Warmup=%d, Medición=%d\n",
           M, N, (double)(M * N * sizeof(float)) / (1024.0 * 1024.0),
           warmup_iters, bench_iters);
    printf("==============================================================================\n");

    size_t num_elements = (size_t)M * N;
    size_t bytes = num_elements * sizeof(float);

    // Host buffers
    float* h_X = (float*)malloc(bytes);
    float* h_Y_online = (float*)malloc(bytes);
    float* h_Y_3pass = (float*)malloc(bytes);
    float* h_Y_cpu = (float*)malloc(bytes);

    if (!h_X || !h_Y_online || !h_Y_3pass || !h_Y_cpu) {
        fprintf(stderr, "[FATAL ERROR] Fallo al alojar memoria en Host.\n");
        return EXIT_FAILURE;
    }

    // Inicialización con valores típicos de logits en transformers/deep learning [-5.0, 5.0]
    srand(12345);
    for (size_t i = 0; i < num_elements; ++i) {
        h_X[i] = ((float)rand() / (float)RAND_MAX) * 10.0f - 5.0f;
    }

    // Device buffers
    float *d_X, *d_Y_online, *d_Y_3pass;
    CUDA_CHECK(cudaMalloc((void**)&d_X, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y_online, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y_3pass, bytes));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

    // Configuración de ejecución del kernel:
    // 1 warp procesa 1 fila.
    // Usamos bloques de 128 hilos = 4 warps por bloque.
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (M + warps_per_block - 1) / warps_per_block;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // --------------------------------------------------------------------------
    // 1. Kernel GPU: Standard 3-Pass Softmax
    // --------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        standard_3pass_softmax_warp_kernel<<<num_blocks, threads_per_block>>>(d_X, d_Y_3pass, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int i = 0; i < bench_iters; ++i) {
        standard_3pass_softmax_warp_kernel<<<num_blocks, threads_per_block>>>(d_X, d_Y_3pass, M, N);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_3pass = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_3pass, ev_start, ev_stop));
    float avg_ms_3pass = ms_3pass / (float)bench_iters;

    // --------------------------------------------------------------------------
    // 2. Kernel GPU: Online Softmax 1-Pass Warp Shfl
    // --------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        online_softmax_warp_kernel<<<num_blocks, threads_per_block>>>(d_X, d_Y_online, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int i = 0; i < bench_iters; ++i) {
        online_softmax_warp_kernel<<<num_blocks, threads_per_block>>>(d_X, d_Y_online, M, N);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_online = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_online, ev_start, ev_stop));
    float avg_ms_online = ms_online / (float)bench_iters;

    // Copia de resultados al Host para validación
    CUDA_CHECK(cudaMemcpy(h_Y_online, d_Y_online, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y_3pass, d_Y_3pass, bytes, cudaMemcpyDeviceToHost));

    // Cálculo de referencia CPU
    printf("[INFO] Calculando referencia Softmax en CPU (double precision)...\n");
    cpu_softmax_reference(h_X, h_Y_cpu, M, N);

    // Verificación numérica
    float max_abs_online = 0.0f, max_rel_online = 0.0f;
    bool ok_online = verify_softmax(h_Y_online, h_Y_cpu, (int)num_elements, &max_abs_online, &max_rel_online);

    float max_abs_3pass = 0.0f, max_rel_3pass = 0.0f;
    bool ok_3pass = verify_softmax(h_Y_3pass, h_Y_cpu, (int)num_elements, &max_abs_3pass, &max_rel_3pass);

    // Tráfico de memoria DRAM efectivo:
    // Lectura de X (bytes) + Escritura de Y (bytes) = 2 * bytes
    double dram_traffic_bytes = 2.0 * (double)bytes;
    double bw_effective_online = (dram_traffic_bytes / (avg_ms_online * 1e-3)) / 1e9;
    double bw_effective_3pass = (dram_traffic_bytes / (avg_ms_3pass * 1e-3)) / 1e9;
    double speedup_online = (double)avg_ms_3pass / (double)avg_ms_online;

    printf("\n| Kernel Softmax | Dimensión (MxN) | Latencia (ms) | BW Efectivo (GB/s) | Speedup vs 3-Pass | Max Error Abs vs CPU | Estado |\n");
    printf("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n");
    printf("| **Standard 3-Pass (Warp Shfl)** | %dx%d | %10.4f | %15.2f | %14.2fx | %19.2e | %s |\n",
           M, N, avg_ms_3pass, bw_effective_3pass, 1.0, max_abs_3pass, ok_3pass ? "PASS [OK]" : "FAIL [ERROR]");
    printf("| **Online 1-Pass (Warp Shfl)**   | %dx%d | %10.4f | %15.2f | %14.2fx | %19.2e | %s |\n",
           M, N, avg_ms_online, bw_effective_online, speedup_online, max_abs_online, ok_online ? "PASS [OK]" : "FAIL [ERROR]");

    // Liberación
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y_online));
    CUDA_CHECK(cudaFree(d_Y_3pass));
    free(h_X);
    free(h_Y_online);
    free(h_Y_3pass);
    free(h_Y_cpu);

    if (!ok_online || !ok_3pass) {
        fprintf(stderr, "\n[FATAL ERROR] Fallo la verificacion numerica contra la referencia Softmax CPU!\n");
        return EXIT_FAILURE;
    }

    printf("\n[SUCCESS] Warp Online Softmax verificado exitosamente con max_abs_error = %.2e < 1e-5.\n",
           max_abs_online);
    return EXIT_SUCCESS;
}
