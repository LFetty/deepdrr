# __device__ float cubictex3d(cudaTextureObject_t tex, float3 coord)
# {
# 	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
# 	const float3 coord_grid = coord - 0.5f;
# 	const float3 index = floor(coord_grid);
# 	const float3 fraction = coord_grid - index;
# 	float3 w0, w1, w2, w3;
#  	const float3 one_frac = 1.0f - fraction;
# 	const float3 squared = fraction * fraction;
# 	const float3 one_sqd = one_frac * one_frac;

# 	w0 = 1.0f/6.0f * one_sqd * one_frac;
# 	w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
# 	w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
# 	w3 = 1.0f/6.0f * squared * fraction;
# 	// WEIGHTS(fraction, w0, w1, w2, w3);

# 	const float3 g0 = w0 + w1;
# 	const float3 g1 = w2 + w3;
# 	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
# 	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]
# 	// fetch the eight linear interpolations
# 	// weighting and fetching is interleaved for performance and stability reasons
# 	float tex000 = tex3D<float>(tex, h0.x, h0.y, h0.z);
# 	float tex100 = tex3D<float>(tex, h1.x, h0.y, h0.z);
# 	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
# 	float tex010 = tex3D<float>(tex, h0.x, h1.y, h0.z);
# 	float tex110 = tex3D<float>(tex, h1.x, h1.y, h0.z);
# 	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
# 	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
# 	float tex001 = tex3D<float>(tex, h0.x, h0.y, h1.z);
# 	float tex101 = tex3D<float>(tex, h1.x, h0.y, h1.z);
# 	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
# 	float tex011 = tex3D<float>(tex, h0.x, h1.y, h1.z);
# 	float tex111 = tex3D<float>(tex, h1.x, h1.y, h1.z);
# 	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
# 	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

# 	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
# }
import cupy as cp
from cupy.cuda import runtime
import numpy as np
import sys

def _get_texture(array:np.ndarray) -> cp.cuda.texture.TextureObject:
    """Get a texture object from a numpy array.

    Args:
        array (np.ndarray): The array to convert to a texture object.

    Returns:
        cupy.cuda.TextureObject: The texture object.
    """
    # Create 3D CUDA array for segmentation
    tex_desc = cp.cuda.texture.TextureDescriptor(addressModes=(runtime.cudaAddressModeClamp, 
                                                               runtime.cudaAddressModeClamp, 
                                                               runtime.cudaAddressModeClamp),
                                         filterMode=runtime.cudaFilterModeLinear,
                                         readMode=runtime.cudaReadModeElementType, 
                                         borderColors=None, 
                                         normalizedCoords=False)
    
    channelformat_desc = cp.cuda.texture.ChannelFormatDescriptor(x=32, 
                                                                 y=0, 
                                                                 z=0, 
                                                                 w=0,
                                                                 f=runtime.cudaChannelFormatKindFloat)
    
    
    arr=cp.asarray(np.moveaxis(array, [0, 1, 2], [2, 1, 0]).copy(), order='C')
    width, height, depth = arr.shape
    
    cuda_array = cp.cuda.texture.CUDAarray(desc=channelformat_desc, 
                                           width=width, 
                                           height=height, 
                                           depth=depth, 
                                           flags=0)
    
    cuda_array.copy_from(np.moveaxis(arr, [0, 1, 2], [2, 1, 0]))
    
    resource_desc = cp.cuda.texture.ResourceDescriptor(restype=runtime.cudaResourceTypeArray, 
                                                       cuArr=cuda_array, 
                                                       #arr=cuda_array, 
                                                       #ChannelFormatDescriptor chDesc=None, 
                                                       #size_t sizeInBytes=0, 
                                                       #size_t width=0, 
                                                       #size_t height=0, 
                                                       )
                    
                    # Create texture object  
    texture = cp.cuda.texture.TextureObject(ResDesc=resource_desc, TexDesc=tex_desc)
    return texture


kernel = """

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
inline __host__ __device__ float2 operator-(float a, float2 b)
{
	return make_float2(a - b.x, a - b.y);
}
// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)  //Danny
{
//    float inv = 1.0f / s;
//    return a * inv;
	return make_float3(s / a.x, s / a.y, s / a.z);
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

inline __host__ __device__ float3 operator-(float a, float3 b)
{
	return make_float3(a - b.x, a - b.y, a - b.z);
}
extern "C"{
__global__ void cubictex3d(cudaTextureObject_t tex, float3 coord, float* out)
{
    //return;
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	const float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
 	const float3 one_frac = 1.0f - fraction;
	const float3 squared = fraction * fraction;
	const float3 one_sqd = one_frac * one_frac;
    
	w0 = 1.0f/6.0f * one_sqd * one_frac;
	w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
	w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
	w3 = 1.0f/6.0f * squared * fraction;
	// WEIGHTS(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]
	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	float tex000 = tex3D<float>(tex, h0.x, h0.y, h0.z);
	float tex100 = tex3D<float>(tex, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	float tex010 = tex3D<float>(tex, h0.x, h1.y, h0.z);
	float tex110 = tex3D<float>(tex, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	float tex001 = tex3D<float>(tex, h0.x, h0.y, h1.z);
	float tex101 = tex3D<float>(tex, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	float tex011 = tex3D<float>(tex, h0.x, h1.y, h1.z);
	float tex111 = tex3D<float>(tex, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction
    
	out[0] = (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction

}
}
"""
imports = """
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "math_functions.h" """

kernel = imports+kernel
print(kernel)
array = np.random.random((128,128,128)).astype(cp.float32)
out = cp.zeros((1), dtype=cp.float32)

position = cp.array([65.,65.,65.], dtype=cp.float32)
texture = _get_texture(array)


mod = cp.RawModule(code=kernel, backend='nvcc')
kernel = mod.get_function('cubictex3d')

kernel.compile(log_stream=sys.stdout)


out2 = kernel(block=(8,8,1), grid=(64,64,1), args=tuple([texture, position, out]))
print(out)