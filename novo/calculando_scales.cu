#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>



#include "ptx.cuh"
#include "utils.cuh"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


using e8m0_t = uint8_t;
using fp8e4m3 = __nv_fp8_e4m3;
typedef __nv_bfloat16 bf16;

constexpr int MXFP8_PREFETCH_BUFFERS_NUM = 0;
constexpr int K = 32;
constexpr int M = 64;
constexpr int BLOCK_SIZE_y = 32;
constexpr int BLOCK_SIZE_x = 64;
constexpr int NUM_THREADS = 64;
constexpr int buffers_num = 2;
constexpr int iterations = 2;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t MXFP8_THREADS_PER_CHUNK = 64;



constexpr size_t MXFP8_CHUNK_DIM_Y = 64;
constexpr size_t MXFP8_ITERATIONS = MXFP8_CHUNK_DIM_Y / BLOCK_SIZE_y;  //   2 = 64 / 32


// Definir quantas iteração teremos por bloco e o formato de cada bloco (MAIOR), já temos os blocos menores 32 x 64

template <typename IType, typename OType>
__global__ void __launch_bounds__(MXFP8_THREADS_PER_CHUNK) cast_mxfp8_col_row_wise(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
    const __grid_constant__ CUtensorMap tensor_map_output_colwise,
    int width,
    int height,
    e8m0_t* scales_colwise,
    e8m0_t* scales_rowwise,
    bool is_col_wise,
    bool is_row_wise,
    bool output_colwise,
    bool output_rowwise
    ) {
        const int global_x = blockIdx.x * BLOCK_SIZE_x; 
        const int global_y = blockIdx.y * BLOCK_SIZE_x; // tem q ser 64 senao vai carregar os mesmos dados da next_iter
        const int tx = threadIdx.x;
        const int warp = threadIdx.x/32;
        __shared__ alignas(128) IType input[buffers_num][BLOCK_SIZE_y][BLOCK_SIZE_x];
        __shared__ alignas(128) OType output_colwise_sh[buffers_num][BLOCK_SIZE_y][BLOCK_SIZE_x];
        __shared__ alignas(128) OType output_rowwise_sh[buffers_num][BLOCK_SIZE_y][BLOCK_SIZE_x];
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ alignas(8) uint64_t mbar[MXFP8_ITERATIONS];
        constexpr int shmem_buff_size = sizeof(input)/ buffers_num;
        const bool is_master_thread = (tx == 0);
        
        ptx::initialize_barriers<MXFP8_ITERATIONS, MXFP8_THREADS_PER_CHUNK>(mbar, is_master_thread);
        // usando TMA carregamos no buffer 0 do input
        ptx::copy_2d_to_shared(&input[0], &tensor_map_input, global_x,
            global_y, shmem_buff_size, &mbar[0],
            is_master_thread);

        int parity = 0;
        #pragma unroll
        for(int iter = 0; iter < iterations; iter++){
            const int next_iter = iter + 1;
            const int atual = iter % 2;
            if(next_iter < iterations){
                const int chunk_offset_x = global_x;
                const int chunk_offset_y = global_y + BLOCK_SIZE_y;
                ptx::copy_2d_to_shared(&input[next_iter % 2], &tensor_map_input, chunk_offset_x,
                    chunk_offset_y, shmem_buff_size, &mbar[next_iter],
                    is_master_thread);

            }
            ptx::fence_proxy_async_shared_cta();

            // Wait for the data to have arrived
            ptx::mbarrier_wait_parity(&mbar[iter], parity);

            if(is_col_wise){
                float max_val = 0.0;
                float elt;
                for(int i = 0; i < 32; i++){
                    elt = fabsf(static_cast<float>(input[atual][i][tx]));
                    max_val = elt > max_val ? elt : max_val;
                }
                const e8m0_t biased_exponent =
                float_to_e8m0(max_val * Quantized_Limits<OType>::max_norm_rcp);

                const float block_scale_inverse = exp2f_rcp(biased_exponent);

                for(int i = 0; i < 32; ++i) {
                    output_colwise_sh[atual][i][tx] =
                        static_cast<OType>(static_cast<float>(input[atual][i][tx]) * block_scale_inverse);
                }

                ptx::fence_proxy_async_shared_cta();
                __syncthreads();

                if(output_colwise){ 
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_x,
                        global_y + (iter * BLOCK_SIZE_y), reinterpret_cast<uint64_t *>(&output_colwise_sh[atual]));

                    // Create a "bulk async-group" out of the previous bulk copy operation.
                    ptx::cp_async_bulk_commit_group();

                    // Wait for TMA transfer to have finished reading shared memory.
                    ptx::cp_async_bulk_wait_group_read<MXFP8_PREFETCH_BUFFERS_NUM>();
                }

                
                int buff = iter % 2;
                int scales_per_col = (BLOCK_SIZE_x * BLOCK_SIZE_x*(height/BLOCK_SIZE_x))/32;
                // suposta correção dos indices
                int dimensao_y = blockIdx.y * ((BLOCK_SIZE_y*BLOCK_SIZE_x)/K) + buff * (BLOCK_SIZE_x/K);
                int dimensao_x = blockIdx.x * scales_per_col + tx;
                scales_colwise[32 * (blockIdx.y + iter) + tx + global_x] = biased_exponent;
                
            }

            if(is_row_wise){
                float max_val = 0.0;
                int dim_x = (threadIdx.x % 2) * 32;
                int dim_y =  threadIdx.x/2;
                float elt;
                for(int i = 0; i < 32; i++){
                    elt = fabsf(static_cast<float>(input[atual][dim_y][dim_x + i]));
                    max_val = elt > max_val ? elt : max_val;
                    
                }
                max_val == 0.0 ? max_val = 1.0 : max_val;
                const e8m0_t biased_exponent =
                float_to_e8m0(max_val * Quantized_Limits<OType>::max_norm_rcp);
                const float block_scale_inverse = exp2f_rcp(biased_exponent);
                for(int j = 0; j < 32; j++){
                    output_rowwise_sh[atual][dim_y][dim_x + j] =
                        static_cast<OType>(static_cast<float>(input[atual][dim_y][dim_x + j]) * block_scale_inverse);
                }
                ptx::fence_proxy_async_shared_cta();
                __syncthreads();
                if(output_rowwise){ 
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_x,
                        global_y + (iter * BLOCK_SIZE_y), reinterpret_cast<uint64_t *>(&output_rowwise_sh[atual]));
                    
                    // Create a "bulk async-group" out of the previous bulk copy operation.
                    ptx::cp_async_bulk_commit_group();

                    // Wait for TMA transfer to have finished reading shared memory.
                    ptx::cp_async_bulk_wait_group_read<MXFP8_PREFETCH_BUFFERS_NUM>();
                }
                int block_tride_y = K / 32;
                // Suposta correção dos indices
                int buff = iter % 2;
                int dimensao_x = blockIdx.x * (BLOCK_SIZE_x/32) * M + (threadIdx.x % 2) * M + iter * BLOCK_SIZE_y;
                int dimensao_y = blockIdx.y * (BLOCK_SIZE_x * (width/32)) + threadIdx.x/2;
                scales_rowwise[(32 * dim_y) + (32 * blockIdx.x) + dim_x + global_y] = biased_exponent; // block_stride_y seria qnt colunas input / 32 (primeira multiplicacao anda intra bloco[0,31], segunda anda extra bloco[0,qnt bloco x], dim_x fala se é o primeiro ou segundo scale intra bloco e global y anda extrabloco nas colunas)
                //blockIdx.x * (BLOCK_SIZE_x/32) * M + (threadIdx.x % 2) * M;
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            
            
        }
        
        ptx::destroy_barriers<MXFP8_ITERATIONS>(mbar, is_master_thread);

    
}

template <typename IType>
__host__ static inline CUtensorMap create_tensor_map(IType* gmem_ptr, int global_height, int global_width, 
    int smem_width, int smem_height) {
    CUtensorMap tensor_map;
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {(uint64_t)global_width, (uint64_t)global_height};
    uint64_t stride[rank - 1] = {(uint64_t)global_width * sizeof(IType)};
    uint32_t box_size[rank] = {(uint32_t)smem_width, (uint32_t)smem_height};
    void* gmem_address = (void*)gmem_ptr;
    uint32_t elem_stride[rank] = {1, 1};
    
    
    if constexpr (std::is_same_v<IType, __nv_bfloat16>){
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gmem_address, size,
        stride, box_size, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    }
    else{
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, gmem_address, size,
        stride, box_size, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    return tensor_map;
}

void quantBF16(torch::Tensor Input, torch::Tensor Output_colwise, torch::Tensor Output_rowwise, torch::Tensor scales_rowwise, torch::Tensor scales_colwise) {
    int M = Input.size(0); // A -> MxK
    int K = Input.size(1);

    // Get data pointers
    bf16* data_ptr_Input = reinterpret_cast<bf16*>(Input.data_ptr<at::BFloat16>());
    fp8e4m3* data_ptr_Output_colwise = reinterpret_cast<fp8e4m3*>(Output_colwise.data_ptr<uint8_t>());
    fp8e4m3* data_ptr_Output_rowwise = reinterpret_cast<fp8e4m3*>(Output_rowwise.data_ptr<uint8_t>());

    e8m0_t* data_ptr_scales_rowwise = scales_rowwise.data_ptr<uint8_t>();
    e8m0_t* data_ptr_scales_colwise = scales_colwise.data_ptr<uint8_t>();

    // Tile sizes
    constexpr int BM = 64;
    constexpr int BK = 64;

    CUtensorMap tensor_map_Input;
    CUtensorMap tensor_map_Output_colwise;
    CUtensorMap tensor_map_Output_rowwise;
    int prev_m = 0;
    int prev_k = 0;

    // Check if we need to reallocate TMA maps
    if (M != prev_m) {
        // Allocate new TMA maps
        tensor_map_Input = create_tensor_map<bf16>(data_ptr_Input, K, M, BM, BK);
        tensor_map_Output_colwise = create_tensor_map<fp8e4m3>(data_ptr_Output_colwise, K, M, BM, BK);
        tensor_map_Output_rowwise = create_tensor_map<fp8e4m3>(data_ptr_Output_rowwise, K, M, BM, BK);
        
        prev_m = M;
        prev_k = K;
    }
    // Assert dimensions are correct
    assert(M == prev_m && K == prev_k);
    // Launch configuration
    dim3 grid((M/BM) * (K/BK));
    dim3 block(NUM_THREADS);

    // Launch kernel
    cast_mxfp8_col_row_wise<bf16, fp8e4m3><<<grid, block>>>(tensor_map_Input, tensor_map_Output_rowwise, tensor_map_Output_colwise, K, M, data_ptr_scales_colwise, data_ptr_scales_rowwise, 1, 1, 1, 1);

}
