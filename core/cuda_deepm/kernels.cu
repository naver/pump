// Copyright 2022-present NAVER Corp.
// CC BY-NC-SA 4.0
// Available only for non-commercial use

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define MAX(x, y)           ((x) < (y) ? (y) : (x))
#define inf std::numeric_limits<float>::infinity()

#define CHECK_CUDA(tensor) {\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory"); \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous"); }
void CHECK_KERNEL() {auto error = cudaGetLastError(); TORCH_CHECK( error == cudaSuccess, cudaGetErrorString(error));}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
#define atomicMax_block atomicMax
#endif


template <typename scalar_t>
__global__ void forward_agg_cuda_kernel( 
        const int LH1, const int LW1, const int LH2, const int LW2, 
        const int gap_left, const int gap_right, float norm,
        const torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> lower,
              torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> upper,
        const float* weights, float* new_weights ) {

    const auto UH1 = LH1 + bool(!gap_left); // level 0 is smaller than other levels
    const auto UW1 = LW1 + bool(!gap_left);
    const auto UH2 = LH2;
    const auto UW2 = LW2;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int uw2 = idx % UW2; idx /= UW2;
    const int uh2 = idx % UH2; idx /= UH2;
    const int uw1 = idx % UW1; idx /= UW1;
    const int uh1 = idx;
    if (uh1 >= UH1) return;

    // then, add the 4 child
    float sumw = 0, nrm = 0, res = 0;
    // #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int v = i/2, u = i%2;
        // source pixel
        const int lh1 = uh1 + (1-v) * gap_left - v * gap_right;
        if (lh1 < 0 || lh1 >= LH1) continue;
        const int lw1 = uw1 + (1-u) * gap_left - u * gap_right;
        if (lw1 < 0 || lw1 >= LW1) continue;

        // load weight even if (lh2,lw2) are invalid
        const float weight = weights ? weights[lh1*LW1 + lw1] : 1;
        sumw += weight;

        const int lh2 = uh2 + 1 - 2*v;
        if (lh2 < 0 || lh2 >= LH2) continue;
        const int lw2 = uw2 + 1 - 2*u;
        if (lw2 < 0 || lw2 >= LW2) continue;

        res += weight * lower[lh1][lw1][lh2][lw2];
        nrm += weight;
    }

    // normalize output
    nrm = sumw * (nrm < sumw ? powf(nrm/sumw, norm) : 1);
    upper[uh1][uw1][uh2][uw2] = (nrm ? res / nrm : 0);
    if (uh2 == 1 && uw2 == 1)
        new_weights[uh1*UW1 + uw1] = sumw;
}

torch::Tensor forward_agg_cuda( int level, float norm, const torch::Tensor lower, 
                                const at::optional<at::Tensor> weights, torch::Tensor upper ) {
    CHECK_CUDA(lower);
    CHECK_CUDA(upper);
    if (weights) CHECK_CUDA(weights.value());

    const auto UH1 = upper.size(0);
    const auto UW1 = upper.size(1);
    const auto UH2 = upper.size(2);
    const auto UW2 = upper.size(3);
    const auto LH1 = lower.size(0);
    const auto LW1 = lower.size(1);
    const auto LH2 = lower.size(2);
    const auto LW2 = lower.size(3);
    TORCH_CHECK( UH1 == LH1 + int(level==1) && UW1 == LW1 + int(level==1), "inconsistent lower and upper shapes" );

    const int gap_left = (level >= 2) ? 1 << (level-2) : 0; // 0, 1, 2, 4, ...
    const int gap_right= 1 << MAX(0, level-2);              // 1, 1, 2, 4, ...

    const int MAX_THREADS = 512; // faster than 1024 (higher SM occupancy)
    const int THREADS_PER_BLOCK = MAX_THREADS;
    const int N_BLOCKS = (UH1*UW1*UH2*UW2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    torch::Tensor new_weights = torch::zeros({UH1, UW1}, upper.options().dtype(torch::kFloat32));

    // one block for each layer, one thread per local-max
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(lower.type(), "forward_agg_cuda", ([&] {
        forward_agg_cuda_kernel<<<N_BLOCKS, THREADS_PER_BLOCK>>>(
            LH1, LW1, LH2, LW2, 
            gap_left, gap_right, norm,
            lower.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            upper.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            weights ? weights->data_ptr<float>() : nullptr, new_weights.data_ptr<float>() );
    }));
    return new_weights;
}

template <typename scalar_t>
__global__ void forward_pool_agg_cuda_kernel( 
        const int LH1, const int LW1, const int LH2, const int LW2, 
        // const int UH1, const int UW1, const int UH2, const int UW2,
        const int gap_left, const int gap_right, float norm,
        const torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> lower,
              torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> upper,
        const float* weights, float* new_weights ) {

    const auto UH1 = LH1 + bool(!gap_left); // level 0 is smaller than other levels
    const auto UW1 = LW1 + bool(!gap_left);
    const auto UH2 = (LH2-1)/2 + 1;
    const auto UW2 = (LW2-1)/2 + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int uw2 = idx % UW2; idx /= UW2;
    const int uh2 = idx % UH2; idx /= UH2;
    const int uw1 = idx % UW1; idx /= UW1;
    const int uh1 = idx;
    if (uh1 >= UH1) return;

    // then, add the 4 child
    float sumw = 0, nrm = 0, res = 0;
    // #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int v = i/2, u = i%2;
        // source pixel
        const int lh1 = uh1 + (1-v) * gap_left - v * gap_right;
        if (lh1 < 0 || lh1 >= LH1) continue;
        const int lw1 = uw1 + (1-u) * gap_left - u * gap_right;
        if (lw1 < 0 || lw1 >= LW1) continue;

        // load weight even if (lh2,lw2) are invalid
        const float weight = weights ? weights[lh1*LW1 + lw1] : 1;
        sumw += weight;

        const int lh2_ = 2*(uh2 + 1 - 2*v); // position in lower
        const int lw2_ = 2*(uw2 + 1 - 2*u);
        float lower_max = -inf;
        #pragma unroll
        for (int j = -1; j <= 1; j++) {
          const int lh2 = lh2_ + j;
          if (lh2 < 0 || lh2 >= LH2) continue;
          #pragma unroll
          for (int i = -1; i <= 1; i++) {
            const int lw2 = lw2_ + i;
            if (lw2 < 0 || lw2 >= LW2) continue;
            float l = lower[lh1][lw1][lh2][lw2];
            lower_max = MAX(lower_max, l);
        }}
        if (lower_max == -inf) continue;

        res += weight * lower_max;
        nrm += weight;
    }

    // normalize output
    nrm = sumw * (nrm < sumw ? powf(nrm/sumw, norm) : 1);
    upper[uh1][uw1][uh2][uw2] = (nrm ? res / nrm : 0);
    if (uh2 == 1 && uw2 == 1)
        new_weights[uh1*UW1 + uw1] = sumw;
}

torch::Tensor forward_pool_agg_cuda( int level, float norm, const torch::Tensor lower, 
                                     const at::optional<at::Tensor> weights, torch::Tensor upper ) {
    CHECK_CUDA(lower);
    CHECK_CUDA(upper);
    if (weights) CHECK_CUDA(weights.value());

    const auto LH1 = lower.size(0);
    const auto LW1 = lower.size(1);
    const auto LH2 = lower.size(2);
    const auto LW2 = lower.size(3);
    const auto UH1 = upper.size(0);
    const auto UW1 = upper.size(1);
    const auto UH2 = upper.size(2);
    const auto UW2 = upper.size(3);
    TORCH_CHECK( UH1 == LH1 + int(level==1) && UW1 == LW1 + int(level==1), "inconsistent lower and upper shapes" );
    TORCH_CHECK( UH2 == (LH2-1)/2+1 && UW2 == (LW2-1)/2+1, "lower level should be twice as big" );

    const int gap_left = (level >= 2) ? 1 << (level-2) : 0; // 0, 1, 2, 4, ...
    const int gap_right= 1 << MAX(0, level-2);              // 1, 1, 2, 4, ...

    const int MAX_THREADS = 512; // faster than 1024 (higher SM occupancy)
    const int THREADS_PER_BLOCK = MAX_THREADS;
    const int N_BLOCKS = (UH1*UW1*UH2*UW2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    torch::Tensor new_weights = torch::zeros({UH1, UW1}, upper.options().dtype(torch::kFloat));
    
    // one block for each layer, one thread per local-max
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(lower.type(), "forward_pool_agg_cuda", ([&] {
        forward_pool_agg_cuda_kernel<<<N_BLOCKS, THREADS_PER_BLOCK>>>(
            LH1, LW1, LH2, LW2, 
            // UH1, UW1, UH2, UW2, 
            gap_left, gap_right, norm,
            lower.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            upper.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            weights ? weights->data<float>() : nullptr, new_weights.data<float>() );
    }));
    return new_weights;
}

__device__ inline int in(int lower, int var, int upper) {
    return lower <= var && var < upper;
}
__device__ inline int sl(bool b) {
    return b ? 1 : -1;
}

__device__ short atomicMaxShort(short* address, short val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3); // multiple of 4
    
    unsigned int order_from[] = {0x0010, 0x0032}; // either bytes[0:2] or bytes[2:4]
    unsigned int from = order_from[((size_t)address & 3) / 2];
    
    unsigned int order_back[] = {0x3254, 0x5410}; // right-to-left 
    unsigned int back = order_back[((size_t)address & 3) / 2];
    unsigned int old, assumed, max_, new_;

    old = *base_address;
    do {
        assumed = old;
        max_ = max(val, (short)__byte_perm(old, 0, from)); // extract word
        new_ = __byte_perm(old, max_, back); // replace word
        old = atomicCAS(base_address, assumed, new_);
    } while (assumed != old);
    return old;
}

template <typename scalar_t>
__device__ inline void TplAtomicMax_block( scalar_t* before, scalar_t after ) { assert(!"atomicMax not implemented for this dtype"); }
template <>
__device__ inline void TplAtomicMax_block( at::Half* before, at::Half after ) { atomicMaxShort( (int16_t*)before, *(int16_t*)&after ); }
template <>
__device__ inline void TplAtomicMax_block( float* before, float after ) { atomicMax_block( (int32_t*)before, *(int32_t*)&after ); }

template <typename scalar_t>
__global__ void backward_agg_unpool_cuda_kernel( 
        const int UH1, const int UW1, 
        const int UH2, const int UW2, 
        const int LH2, const int LW2, 
        const int gap_left, const int gap_right,
        const torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> upper,
              torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> lower ) {

    /* Each block is going to take care of a single layer, i.e. lower[:,:,0::2,0::2].
       the first thread is allocating some global memory and then frees it later.
    */
    // const int LH1 = gridDim.x;
    // const int LW1 = gridDim.y;
    const int lh1 = blockIdx.y;
    const int lw1 = blockIdx.x;
    const int UHW2 = UH2 * UW2; // upper layer size

    __shared__ float* _shared_addr;
    if (threadIdx.x == 0)
        do{ _shared_addr = new float [2*UHW2]; } // for each upper place, we have (best, bestp)
        while(!_shared_addr); // waiting for memory to be available...
    __syncthreads();

    float * layer_best = _shared_addr;
    int * layer_bestp = (int*)(_shared_addr+1); //UHW);
    assert( layer_best );

    /* First pass: we recover the position and values of all local maxima in the layer
    */ 
    for (int idx = threadIdx.x; idx < UHW2; idx += blockDim.x) {
        const int ux = idx % UW2;
        const int uy = idx / UW2;
        const int lx = 2*ux; // lower pos from upper pos
        const int ly = 2*uy;

        // argmax my local minima
        float best = -inf;
        int bestp = 0;
        #pragma unroll
        for (int j_= -1; j_<= 1; j_++) {
          const int j = ly + j_;
          if (j < 0 || j >= LH2) continue;
          #pragma unroll
          for (int i_= -1; i_<= 1; i_++) {
            const int i = lx + i_;
            if (i < 0 || i >= LW2) continue;
            float cur = lower[lh1][lw1][j][i];
            if (cur > best) { best = cur; bestp = j*LW2+i; }
        }}
        layer_best[2*idx] = best;
        layer_bestp[2*idx] = bestp;
    }
    
    __syncthreads();
    
    /* Second pass: we update the local maxima according to the upper layer
    */ 
    for (int idx = threadIdx.x; idx < UHW2; idx += blockDim.x) {
        const int ux = idx % UW2;
        const int uy = idx / UW2;

        // max-pool the additional value from the upper layer
        scalar_t add = 0;
        for (int v = -gap_left; v <= gap_right; v += gap_right+gap_left) {
          for (int u = -gap_left; u <= gap_right; u += gap_right+gap_left) {
            const int uh1 = lh1 + v, uw1 = lw1 + u;
            const int uh2 = uy+sl(v>0), uw2 = ux+sl(u>0);
            if (in(0, uh1, UH1) && in(0, uw1, UW1) && in(0, uh2, UH2) && in(0, uw2, UW2))
                add = MAX(add, upper[uh1][uw1][uh2][uw2]);
        }}

        // grab local maxima
        float best = layer_best[2*idx];
        int bestp = layer_bestp[2*idx];
        const int lx = bestp % LW2;
        const int ly = bestp / LW2;

        // printf("UH=%d,UW=%d: uy=%d,ux=%d --> best=%g at ly=%d,lx=%d\n", UH,UW, uy,ux, best, ly,lx);
        scalar_t* before = & lower[lh1][lw1][ly][lx];
        scalar_t  after  = best + add;
        TplAtomicMax_block<scalar_t>( before, after );
    }

    __syncthreads();

    if (threadIdx.x == 0) 
        delete _shared_addr;
}

void backward_agg_unpool_cuda( int level, const torch::Tensor upper, torch::Tensor lower, bool exclude_borders ) {
    CHECK_CUDA(lower);
    CHECK_CUDA(upper);

    const auto UH1 = upper.size(0);
    const auto UW1 = upper.size(1);
    const auto UH2 = upper.size(2);
    const auto UW2 = upper.size(3);
    const auto LH1 = lower.size(0);
    const auto LW1 = lower.size(1);
    const auto LH2 = lower.size(2);
    const auto LW2 = lower.size(3);
    TORCH_CHECK( UH1 == LH1 + int(level==1) && UW1 == LW1 + int(level==1), "inconsistent lower and upper shapes" );
    const int xb = exclude_borders; // local_argmax cannot reach the bottom and right borders

    const int gap_left = (level >= 2) ? 1 << (level-2) : 0; // 0, 1, 2, 4, ...
    const int gap_right= 1 << MAX(0, level-2);              // 1, 1, 2, 4, ...

    const int64_t MAX_THREADS = 1024;
    const int64_t THREADS_PER_LAYER = MIN(UH2*UW2, MAX_THREADS);

    // one block for each layer, one thread per local-max
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(upper.type(), "backward_agg_unpool_cuda", ([&] {
        backward_agg_unpool_cuda_kernel<<<dim3(LW1,LH1), THREADS_PER_LAYER>>>(
            UH1, UW1, UH2, UW2, LH2-xb, LW2-xb, 
            gap_left, gap_right,
            upper.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            lower.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>());
    }));
    CHECK_KERNEL();
}

template <typename scalar_t>
__global__ void max_pool3d_cuda_kernel( 
        const int BS, const int NC, const int IH, const int IW, const int OH, const int OW, 
        const int ks, const int stride,
        const torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> tensor,
              torch::PackedTensorAccessor64<scalar_t,3,torch::RestrictPtrTraits> maxima,
              torch::PackedTensorAccessor64<int64_t,    3,torch::RestrictPtrTraits> indices ) {

    // each thread takes care of one output
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = idx % OW; idx /= OW;
    const int y = idx % OH; idx /= OH;
    const int b = idx;
    if (b >= BS) return;

    float best = -inf;
    int64_t best_pos = 0;
    for (int64_t c = 0; c < NC; c++) {
      for (int j = stride*y; j < stride*y+ks; j++) {
        for (int i = stride*x; i < stride*x+ks; i++) {
            // assert( b < BS and c < NC and j < IH and i < IW );
            float cur = tensor[b][c][j][i];
            if (cur > best) {best = cur; best_pos = (c*IH + j)*IW+ i; }
    }}}

    // assert( b < BS and y < OH and x < OW );
    maxima [b][y][x] = best;
    indices[b][y][x] = best_pos;
}

void max_pool3d_cuda( const torch::Tensor tensor, const int kernel_size, const int stride,
                            torch::Tensor maxima, torch::Tensor indices ) {
    CHECK_CUDA(tensor);
    TORCH_CHECK(tensor.dim() == 4, "tensor should be 4-dimensional: BxCxHxW");
    const int BS = tensor.size(0);
    const int NC = tensor.size(1);
    const int IH = tensor.size(2); // input height
    const int IW = tensor.size(3); // input width

    // output size
    TORCH_CHECK( maxima.sizes() == indices.sizes(), "maxima and indices should have the same shape" );
    TORCH_CHECK( BS == maxima.size(0), "bad batch size" );
    const int OH = maxima.size(1);
    const int OW = maxima.size(2);

    const int64_t THREADS_PER_LAYER = 512;
    const int64_t N_BLOCKS = (BS*OH*OW + THREADS_PER_LAYER-1) / THREADS_PER_LAYER;
    
    // one block for each layer, one thread per local-max
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.type(), "max_pool3d_cuda", ([&] {
       max_pool3d_cuda_kernel<<<N_BLOCKS, THREADS_PER_LAYER>>>(
            BS, NC, IH, IW, OH, OW, kernel_size, stride,
            tensor. packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
            maxima. packed_accessor64<scalar_t,3,torch::RestrictPtrTraits>(),
            indices.packed_accessor64<int64_t,3,torch::RestrictPtrTraits>());
    }));
}


__device__ inline float ptdot( const float* m, float x, float y ) {
  return x*m[0] + y*m[1] + m[2];
}

__device__ inline float sqr(float v) {
    return v*v;
}


__global__ void merge_corres_cuda_kernel( 
            const int OH, const int OW, const int OZ, const int IH, const int IW, 
            const float dmax2, int offset, const float* inv_rot, const int all_step,
            const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> corres_a,
                  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> all_corres_a ) {

    // each thread takes care of one output
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx % OW; idx /= OW;
    const int j = idx;
    if (j >= OH) return;

    const float tol2 = 2*2; // squared
    auto all_cor = all_corres_a[j][i];
    
    // center of the bin in the reference frame
    float x = i*all_step + all_step/2;
    float y = j*all_step + all_step/2;

    // center of the bin on the rescaled+rotated image
    float xr = ptdot( inv_rot + 0, x, y ); 
    float yr = ptdot( inv_rot + 3, x, y );

    // iterate on the nearby bins
    int xb = (int)(0.5+ xr/4); // rescaled+rotated desc always has step 4
    int yb = (int)(0.5+ yr/4);
    
    float best = dmax2;
    #pragma unroll
    for (int _v = -1; _v <= 1; _v++) {
      #pragma unroll
      for (int _u = -1; _u <= 1; _u++) {
        const int v = yb+_v, u = xb+_u;
        if (!(in(0, v, IH) && in(0, u, IW))) continue;
        auto cor = corres_a[v][u];
        float d = sqr(cor[offset]-x) + sqr(cor[offset+1]-y);
        if (d < best)  best = d;
    }}

    #pragma unroll
    for (int _v = -1; _v <= 1; _v++) {
      #pragma unroll
      for (int _u = -1; _u <= 1; _u++) {
        const int v = yb+_v, u = xb+_u;
        if (!(in(0, v, IH) && in(0, u, IW))) continue;
        auto cor = corres_a[v][u];
        float d = sqr(cor[offset]-x) + sqr(cor[offset+1]-y);
        if (d <= tol2*best) { // spatially close
            // merge correspondence if score is better than actual
            if (cor[4] > all_cor[4])
              for (int k = 0; k < OZ; k++) all_cor[k] = cor[k];
          }
    }}
}

void merge_corres_cuda( const torch::Tensor corres, const int offset, const torch::Tensor _inv_rot, 
                        const float dmax, torch::Tensor all_corres, const int all_step ) {
    CHECK_CUDA( corres );
    CHECK_CUDA( all_corres );
    CHECK_CUDA( _inv_rot );
    TORCH_CHECK(_inv_rot.is_contiguous(), "inv_rot should be contiguous" );

    const int IH = corres.size(0);
    const int IW = corres.size(1);
    const int IZ = corres.size(2);
    const int OH = all_corres.size(0);
    const int OW = all_corres.size(1);
    const int OZ = all_corres.size(2);
    TORCH_CHECK( IZ == OZ, "corres and all_corres should have the same shape[2]" );

    const int THREADS_PER_LAYER = 512;
    const int N_BLOCKS = (OH * OW + THREADS_PER_LAYER-1) / THREADS_PER_LAYER;
    
    merge_corres_cuda_kernel<<<N_BLOCKS, THREADS_PER_LAYER>>>(
        OH, OW, OZ, IH, IW, dmax*dmax, offset, _inv_rot.data_ptr<float>(), all_step,
                corres.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            all_corres.packed_accessor32<float,3,torch::RestrictPtrTraits>());
    CHECK_KERNEL();
}


template <typename scalar_t>
__global__ void mask_correlations_radial_cuda_kernel( 
            float radius, const float alpha,
            const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> targets,
                  torch::PackedTensorAccessor64<scalar_t,4,torch::RestrictPtrTraits> corr ) {

    #define H1 ((int)corr.size(0))
    #define W1 ((int)corr.size(1))
    #define H2 ((int)corr.size(2))
    #define W2 ((int)corr.size(3))

    // each block takes care of one layer corr[j,i,:,:]
    const int j = blockIdx.x / W1;
    const int i = blockIdx.x % W1;
    if (j >= H1) return;

    // read the target center
    const float cx = targets[j][i][0];
    const float cy = targets[j][i][1];
    if (cx != cx || cy != cy) return; // undefined center
    radius *= radius; // squared
    const float alpha_out = (alpha > 1 ? 1 : alpha);
    const float alpha_in = (alpha < 1 ? 1 : alpha);
    
    for (int idx = threadIdx.x; idx < H2*W2; idx += blockDim.x) {
        const int v = idx / W2;
        const int u = idx % W2;

        // compute weighting
        float dis2 = sqr(u - cx) + sqr(v - cy);
        float mul = alpha_in;
        if (dis2 > radius) 
            mul = 1 - alpha_out*(1 - radius / dis2);

        corr[j][i][v][u] *= mul; 
    }
}

void mask_correlations_radial_cuda( torch::Tensor corr, const torch::Tensor targets, 
                                    const float radius, const float alpha) {
    CHECK_CUDA( corr );
    CHECK_CUDA( targets );

    const int THREADS_PER_LAYER = 512;
    const int N_BLOCKS = H1*W1;

    #undef H1
    #undef W1
    #undef H2
    #undef W2

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(corr.type(), "mask_correlations_radial_cuda", ([&] {    
        mask_correlations_radial_cuda_kernel<<<N_BLOCKS, THREADS_PER_LAYER>>>(
            radius, alpha,
            targets.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
               corr.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>());
    }));
    CHECK_KERNEL();
}
