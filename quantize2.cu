#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;

using namespace std;
#define FINAL_MASK 0xffffffff  // Máscara completa para todos os threads em um warp
__global__ void kernelExample(half* globalMat, cutlass::float_ue8m0_t *scale, int n, int m, int N, int M)
{
    // Índice do thread atual
    int indice = threadIdx.x + threadIdx.y * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int warp = indice / 32;
    int num_warps = (blockDim.x * blockDim.y + 31) / 32;
    int posicao;
    half local = __float2half(0.0f);  // Inicialização com zero
    // Verifica se o índice está dentro dos limites da matriz
    if (idx < N && idy < M) {
        local = globalMat[idx + idy * N];
    }
    __syncwarp();
    for(int j = 0; j < n/num_warps; j++) {
        posicao = warp + num_warps * j + blockIdx.x * blockDim.y + blockIdx.y * blockDim.x + blockIdx.y * (N + n - 1)/n;
        local = globalMat[idx + idy * N];
        half maxVal = __habs(local);
        for (int offset = 16; offset > 0; offset /= 2) {
            half shuffled = __shfl_xor_sync(FINAL_MASK, maxVal, offset, 32);
            maxVal = __hmax(__habs(maxVal), __habs(shuffled));
        }
        __syncwarp();
        if (threadIdx.x % 32 == 0) {
            // Convertendo para float para operações aritméticas
            float maxVal_f = __half2float(maxVal);
            if (maxVal_f != 0.0f) {  // Evitando divisão por zero
                float scale_value= 225.0f / fabsf(maxVal_f);
                // Usando o construtor explícito
                cutlass::float_ue8m0_t e8m0_value;
                // Inicializando com um valor de ponto flutuante usando o método de conversão do CUTLASS
                cutlass::NumericConverter<cutlass::float_ue8m0_t, float> converter;
                e8m0_value = converter(scale_value);
                // Atribuindo ao array de saída
                scale[posicao] = e8m0_value;
            } else {
                // Para zero, usamos o construtor com valor zero
                cutlass::float_ue8m0_t zero_value;
                cutlass::NumericConverter<cutlass::float_ue8m0_t, float> converter;
                zero_value = converter(0.0f);
                scale[posicao] = zero_value;
            }
        }
        __syncwarp();
    }
}
// Função de host para configurar e lançar o kernel
void launchKernelExample(half* d_globalMat, cutlass::float_ue8m0_t *scale, int n, int m, int N, int M) {
    // Definindo configuração de blocos e grid
    dim3 threadsPerBlock(m, n);  // 256 threads por bloco
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // Lançando o kernel
    kernelExample<<<numBlocks, threadsPerBlock>>>(d_globalMat, scale, n, m, N, M);
    // Verificando erros de lançamento
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro ao lançar kernel: %s\n", cudaGetErrorString(err));
    }
}
// Função principal
//na função principal, inicializamos a matriz global e o vetor de escala, alocamos memória na GPU e chamamos a função de lançamento do kernel
void scales(torch::Tensor torch_tensor) {
    
    
    // Dimensões da matriz
    auto sizes = torch_tensor.sizes(); // shape da matriz
    int M = sizes[0];  // Número de linhas
    int N = sizes[1];  // Número de colunas
    int n = 32;    // Dimensão do bloco em x
    int m = 32;    // Dimensão do bloco em y
    
    
    // Vetores Thrust para escalas
    thrust::host_vector<cutlass::float_ue8m0_t> h_scale(N * M/32);
    thrust::device_vector<cutlass::float_ue8m0_t> d_scale_thrust(N * M/32);
    
    torch::Half* torch_data_ptr = torch_tensor.data_ptr<torch::Half>();
    half* d_globalMat = reinterpret_cast<half*>(torch_data_ptr);


    // Obtendo ponteiro raw do vetor Thrust para usar no kernel
    cutlass::float_ue8m0_t* d_scale_raw = thrust::raw_pointer_cast(d_scale_thrust.data());
    
    // Lançando o kernel com o ponteiro raw
    launchKernelExample(d_globalMat, d_scale_raw, n, m, N, M);
    
    // Copiando de volta do device para o host usando Thrust
    // Isso é uma cópia automática - não é necessário cudaMemcpy
    h_scale = d_scale_thrust;
    
    for(int i = 0; i < N * M / 32; i++) {
        cout << "Scale[" << i << "] = " << h_scale[i] << endl;
    }

    // Liberação de memória
    cudaFree(d_globalMat);
    // Não precisa liberar os vetores Thrust manualmente
    // Eles são liberados automaticamente quando saem de escopo

    //return h_scale;
}

PYBIND11_MODULE(neoQuant, m) {
    m.doc() = "Quantize FP16 to MXFP8";
    
    m.def("scales", &scales, 
          "Return the scales to dequantize a torch MXFP8 tensor to FP16",
          py::arg("torch_tensor"));
  }

