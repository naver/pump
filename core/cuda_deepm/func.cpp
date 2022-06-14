// Copyright 2022-present NAVER Corp.
// CC BY-NC-SA 4.0
// Available only for non-commercial use

#include <torch/extension.h>
using namespace torch::indexing; // Slice
#include <vector>

#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define MAX(x, y)           ((x) < (y) ? (y) : (x))
#define CHECK_CUDA(x)       TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline Slice sl(bool x) {
    if (x)
        return Slice(0, -1);
    else
        return Slice(1, None);
}

torch::Tensor forward_agg_cuda( int level, float norm, const torch::Tensor lower, 
                                const at::optional<at::Tensor> weights, torch::Tensor upper );

std::vector<torch::Tensor> forward_agg( int level, float norm, const torch::Tensor lower, 
                                        const at::optional<at::Tensor> weights = at::nullopt ) {
    TORCH_CHECK(level >= 1, "level must be >= 1");
    TORCH_CHECK(lower.dim() == 4, "input must have 4 dimensions");
    const auto LH1 = lower.size(0);
    const auto LW1 = lower.size(1);
    const auto LH2 = lower.size(2);
    const auto LW2 = lower.size(3);
    if (weights) TORCH_CHECK(weights->size(0) == LH1 && weights->size(1) == LW1, "weights should have shape == lower.shape[:2]");
    const auto UH1 = (level == 1) ? LH1+1 : LH1;
    const auto UW1 = (level == 1) ? LW1+1 : LW1;

    TORCH_CHECK(lower.is_cuda())
    auto upper = torch::zeros({UH1, UW1, LH2, LW2}, lower.options());
    torch::Tensor new_weights = forward_agg_cuda( level, norm, lower, weights, upper );
    return {upper, new_weights};
}


torch::Tensor forward_pool_agg_cuda( int level, float norm, const torch::Tensor lower,
                                     const at::optional<at::Tensor> weights, torch::Tensor upper );

std::vector<torch::Tensor> forward_pool_agg( int level, float norm, const torch::Tensor lower, 
                                        const at::optional<at::Tensor> weights = at::nullopt ) {
    TORCH_CHECK(level >= 1, "level must be >= 1");
    TORCH_CHECK(lower.dim() == 4, "input must have 4 dimensions");
    const auto LH1 = lower.size(0);
    const auto LW1 = lower.size(1);
    const auto LH2 = lower.size(2);
    const auto LW2 = lower.size(3);
    if (weights) TORCH_CHECK(weights->size(0) == LH1 && weights->size(1) == LW1, "weights should have shape == lower.shape[:2]");
    const auto UH1 = (level == 1) ? LH1+1 : LH1;
    const auto UW1 = (level == 1) ? LW1+1 : LW1;

    TORCH_CHECK(lower.is_cuda())
    auto upper = torch::zeros({UH1, UW1, 1+(LH2-1)/2, 1+(LW2-1)/2}, lower.options());
    torch::Tensor new_weights = forward_pool_agg_cuda( level, norm, lower, weights, upper );
    return {upper, new_weights};
}

// forward declaration
void backward_agg_unpool_cuda( int level, const torch::Tensor upper, torch::Tensor lower, bool exclude_borders );

void backward_agg_unpool( int level, const torch::Tensor upper, torch::Tensor lower, bool exclude_borders = true ) {
    TORCH_CHECK(level >= 1, "level must be >= 1");
    TORCH_CHECK( upper.dim() == 4 && lower.dim() == 4, "inputs should be 4-dimensional" );

    TORCH_CHECK(upper.is_cuda() && lower.is_cuda())
    backward_agg_unpool_cuda(level, upper, lower, exclude_borders);
}


void max_pool3d_cuda( const torch::Tensor tensor, const int kernel_size, const int stride,
                            torch::Tensor maxima, torch::Tensor indices );

std::vector<torch::Tensor> max_pool3d( const torch::Tensor tensor, const int kernel_size, const int stride ) {
    TORCH_CHECK(tensor.dim() == 4, "tensor should be 4-dimensional: BxCxHxW");
    TORCH_CHECK( 1 <= kernel_size, "bad kernel size %d", kernel_size );
    TORCH_CHECK( 1 <= stride, "bad stride %d", stride );
    const int IB = tensor.size(0);
    const int IH = tensor.size(2); // input height
    const int IW = tensor.size(3); // input width

    // output size
    const int OH = 1 + (IH - kernel_size) / stride;
    const int OW = 1 + (IW - kernel_size) / stride;
    
    torch::Tensor maxima  = torch::empty({IB, OH, OW}, tensor.options());
    torch::Tensor indices = torch::empty({IB, OH, OW}, tensor.options().dtype(torch::kInt64));

    if (tensor.is_cuda())
        max_pool3d_cuda( tensor, kernel_size, stride, maxima, indices );
    else
        TORCH_CHECK(false, "CPU max_pool3d not implemented yet");
    return {maxima, indices};
}

static inline float ptdot( const float* m, float x, float y ) {
  return x*m[0] + y*m[1] + m[2];
}

static inline float pow2(float v) {
    return v*v;
}

void merge_corres_cpu( const torch::Tensor corres, int offset, const torch::Tensor _inv_rot, 
                       float dmax, torch::Tensor all_corres, const int all_step ) {
    const int H = corres.size(0);
    const int W = corres.size(1);
    const float tol = 2*2; // squared
    dmax *= dmax; // squared

    TORCH_CHECK( _inv_rot.is_contiguous() );
    const float* inv_rot = _inv_rot.data_ptr<float>();

    auto corres_a = corres.accessor<float,3>();
    auto all_corres_a = all_corres.accessor<float,3>();

    // for each bin of the final histograms, we get the nearest-neighbour bin in corres0 and corres1
    for (int j=0; j<all_corres.size(0); j++) 
      for (int i=0; i<all_corres.size(1); i++) {
        // printf("accessing all_corres[%d,%d]", j, i);
        auto all_cor = all_corres_a[j][i];
        
        // center of the bin in the reference frame
        float x = i*all_step + all_step/2;
        float y = j*all_step + all_step/2;
        // printf(" -> (%g,%g) in ref img", x, y);

        // center of the bin on the rescaled+rotated image
        float xr = ptdot( inv_rot + 0, x, y ); 
        float yr = ptdot( inv_rot + 3, x, y );
        // printf(" -> (%g,%g) in rescaled", xr, yr);

        // iterate on the nearby bins
        int xb = (int)(0.5+ xr/4); // rescaled+rotated desc always has step 4
        int yb = (int)(0.5+ yr/4);
        // printf(" -> (%d,%d) in bins\n", xb, yb);

        float best = dmax;
        for (int v = MAX(0,yb-1); v <= MIN(H,yb+1); v++)
          for (int u = MAX(0,xb-1); u <= MIN(W,xb+1); u++) {
            // assert( v >= 0 && v < corres_a.size(0) );
            // assert( u >= 0 && u < corres_a.size(1) );
            auto cor = corres_a[v][u];
            float d = pow2(cor[offset]-x) + pow2(cor[offset+1]-y);
            if( d < best )  best = d;
        }

        for (int v = MAX(0,yb-1); v <= MIN(H,yb+1); v++)
          for (int u = MAX(0,xb-1); u <= MIN(W,xb+1); u++) {
            // assert( v >= 0 && v < corres_a.size(0) );
            // assert( u >= 0 && u < corres_a.size(1) );
            auto cor = corres_a[v][u];
            float d = pow2(cor[offset]-x) + pow2(cor[offset+1]-y);
            if( d <= tol*best ) { // spatially close
                // merge correspondence if score is better than actual
                // printf("update all_corres[%d,%d]\n", v,u);
                if( cor[4] > all_cor[4] )
                  for (int k = 0; k < all_corres.size(2); k++) 
                    all_cor[k] = cor[k];
              }
        }
    }
}

void merge_corres_cuda( const torch::Tensor corres, int offset, const torch::Tensor inv_rot, 
                        float dmax, torch::Tensor all_corres, const int all_step );

void merge_corres( const torch::Tensor corres, int offset, const torch::Tensor rot, 
                   torch::Tensor all_corres, const int all_step ) {
    TORCH_CHECK(     corres.dim() == 3 &&     corres.size(2) == 6,     "corres.shape should be (H,W,6)" );
    TORCH_CHECK( all_corres.dim() == 3 && all_corres.size(2) == 6, "all_corres.shape should be (H,W,6)" );

    float dmax = 8 * torch::sqrt(torch::det(rot)).item<float>();
    torch::Tensor inv_rot = torch::inverse(rot).contiguous();

    if (all_corres.is_cuda()) 
        merge_corres_cuda( corres, offset, inv_rot, dmax, all_corres, all_step );
    else
        merge_corres_cpu( corres, offset, inv_rot, dmax, all_corres, all_step );
}


void mask_correlations_radial_cuda( torch::Tensor corr, const torch::Tensor targets, 
                                    const float radius, const float alpha);

void mask_correlations_radial( torch::Tensor corr, const torch::Tensor targets, 
                                    const float radius, const float alpha) {
    // radius: protected area in pixels around each target center
    // alpha: in [0,1]. If alpha = 0: no effect. If alpha = 1: full effect.
    TORCH_CHECK( corr.dim() == 4 );
    TORCH_CHECK( targets.dim() == 3 );
    TORCH_CHECK( targets.size(0) == corr.size(0) && targets.size(1) == corr.size(1) && targets.size(2) == 2, 
        "correlations and targets should have the same shape[:2]" );

    if (corr.is_cuda()) 
        mask_correlations_radial_cuda( corr, targets, radius, alpha );
    else
        TORCH_CHECK(false, "TODO");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_agg", &forward_agg, "forward aggregation (CUDA)");
  m.def("forward_pool_agg", &forward_pool_agg, "forward pooling and aggregation (CUDA)");
  m.def("backward_agg_unpool", &backward_agg_unpool, "backward sparse-conv and max-unpooling (C++ & CUDA)");
  m.def("max_pool3d", &max_pool3d, "max_pool3d that can handle big inputs (CUDA)");
  m.def("merge_corres_one_side", &merge_corres, "merge correspondences on CPU or GPU" );
  m.def("mask_correlations_radial", &mask_correlations_radial, "mask correlations radially (CUDA)" );
}
