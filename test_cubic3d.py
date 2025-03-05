import cupy as cp

kernel = """"
template<class floatN, class T>
__device__ floatN CUBICTEX3D(cudaTextureObject_t tex, float3 coord)
{
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	const float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
	WEIGHTS(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]
	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	floatN tex000 = tex3D<T>(tex, h0.x, h0.y, h0.z);
	floatN tex100 = tex3D<T>(tex, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	floatN tex010 = tex3D<T>(tex, h0.x, h1.y, h0.z);
	floatN tex110 = tex3D<T>(tex, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	floatN tex001 = tex3D<T>(tex, h0.x, h0.y, h1.z);
	floatN tex101 = tex3D<T>(tex, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	floatN tex011 = tex3D<T>(tex, h0.x, h1.y, h1.z);
	floatN tex111 = tex3D<T>(tex, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
}
"""

cp.RawKernel(source=kernel, )