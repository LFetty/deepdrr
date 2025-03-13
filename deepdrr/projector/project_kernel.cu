#include <cubicTex3D.cu>
#include <stdio.h>

// Supports at most 20 volumes.

#define SEG_PASTER(vol_id, mat_id) seg_##vol_id##_##mat_id
#define SEG(vol_id, mat_id) SEG_PASTER(vol_id, mat_id)
#define VOL_PASTER(vol_id) volume_##vol_id
#define VOLUME(vol_id) VOL_PASTER(vol_id)

#define SEG_TEX(vol_id, mat_id) seg_tex[vol_id * NUM_MATERIALS + mat_id]
#define VOL_TEX(vol_id) volume_tex[vol_id]

// __device__ cudaTextureObject_t volume_tex;
// __device__ cudaTextureObject_t seg_tex;

#ifndef NUM_MATERIALS
#define NUM_MATERIALS 14
#endif

#ifndef NUM_VOLUMES
#define NUM_VOLUMES 1
#endif

#ifndef ATTENUATE_OUTSIDE_VOLUME
#define ATTENUATE_OUTSIDE_VOLUME 0
#endif

#ifndef AIR_DENSITY
#define AIR_DENSITY 0.1129
#endif

#ifndef AIR_INDEX
#define AIR_INDEX 0
#endif

/*** Handle one volume ***/
// #if NUM_VOLUMES > 0
// #define CURR_VOL_ID 0
// // the CT volume
// texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// // channel of the materials array, same size as the volume.
// #if NUM_MATERIALS > 0
// texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
// #endif
// #if NUM_MATERIALS > 1
// texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
// #endif
// #if NUM_MATERIALS > 1
// texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
// #endif

// #undef CURR_VOL_ID
// #endif

#define UPDATE(multiplier, vol_id, mat_id)                                     \
  do {                                                                         \
    area_density[(mat_id)] += (multiplier)*tex3D<float>(VOL_TEX(vol_id), px[vol_id],   \
                                                 py[vol_id], pz[vol_id]) *     \
                              seg_at_alpha[vol_id][mat_id];                    \
  } while (0)

#define GET_POSITION_FOR_VOL(vol_id)                                           \
  do {                                                                         \
    /* Get the current sample point in the volume voxel-space. */              \
    /* In CUDA, voxel centers are located at (xx.5, xx.5, xx.5), whereas       \
     * SwVolume has voxel centers at integers. */                              \
    px[vol_id] = sx_ijk[vol_id] + alpha * rx_ijk[vol_id] - 0.5;                \
    py[vol_id] = sy_ijk[vol_id] + alpha * ry_ijk[vol_id] - 0.5;                \
    pz[vol_id] = sz_ijk[vol_id] + alpha * rz_ijk[vol_id] - 0.5;                \
  } while (0)

#define LOAG_SEGS_FOR_VOL_MAT(vol_id, mat_id)                                  \
  do {                                                                         \
    seg_at_alpha[vol_id][mat_id] = round(                                      \
        tex3D<float>(SEG_TEX(vol_id, mat_id), px[vol_id], py[vol_id], pz[vol_id]));  \  
  } while (0)

// TODO: rather than having num vols lines for each macro, define the macro once
// with #if statements for each vol_id.
#if NUM_MATERIALS == 1
#define LOAD_SEGS_FOR_VOL(vol_id)                                              \
  do {                                                                         \
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);                                          \
  } while (0)
#elif NUM_MATERIALS == 2
#define LOAD_SEGS_FOR_VOL(vol_id)                                              \
  do {                                                                         \
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);                                          \
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);                                          \
  } while (0)
#elif NUM_MATERIALS == 3
  #define LOAD_SEGS_FOR_VOL(vol_id)                                              \
    do {                                                                         \
      LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);                                          \
      LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);                                          \
      LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);                                          \
    } while (0)
#else
#define LOAD_SEGS_FOR_VOL(vol_id)                                              \
  do {                                                                         \
    printf("NUM_MATERIALS not in [1, 14]");                                    \
  } while (0)
#endif

#if NUM_VOLUMES == 1
#define LOAD_SEGS_AT_ALPHA                                                     \
  do {                                                                         \
    if (do_trace[0]) {                                                         \
      GET_POSITION_FOR_VOL(0);                                                 \
      LOAD_SEGS_FOR_VOL(0);                                                    \
    }                                                                          \
  } while (0)
#else
#define LOAD_SEGS_AT_ALPHA                                                     \
  do {                                                                         \
    printf(                                                                    \
        "LOAD_SEGS_AT_ALPHA not supported for NUM_VOLUMES outside [1, 20]");   \
  } while (0)
#endif

#if NUM_MATERIALS == 1
#define INTERPOLATE_FOR_VOL(multiplier, vol_id)                                \
  do {                                                                         \
    UPDATE(multiplier, vol_id, 0);                                             \
  } while (0)
#elif NUM_MATERIALS == 2
#define INTERPOLATE_FOR_VOL(multiplier, vol_id)                                \
  do {                                                                         \
    UPDATE(multiplier, vol_id, 0);                                             \
    UPDATE(multiplier, vol_id, 1);                                             \
  } while (0)
#elif NUM_MATERIALS == 3
  #define INTERPOLATE_FOR_VOL(multiplier, vol_id)                                \
    do {                                                                         \
      UPDATE(multiplier, vol_id, 0);                                             \
      UPDATE(multiplier, vol_id, 1);                                             \
      UPDATE(multiplier, vol_id, 2);                                             \
    } while (0)
#else
#define INTERPOLATE_FOR_VOL(multiplier, vol_id)                                \
  do {                                                                         \
    printf("NUM_MATERIALS not in [1, 14]");                                    \
  } while (0)
#endif

#if NUM_VOLUMES == 1
#define INTERPOLATE(multiplier)                                                \
  do {                                                                         \
    if (do_trace[0] && (priority[0] == curr_priority)) {                       \
      INTERPOLATE_FOR_VOL(multiplier, 0);                                      \
    }                                                                          \
  } while (0)
#else
#define INTERPOLATE(multiplier)                                                \
  do {                                                                         \
    printf("INTERPOLATE not supported for NUM_VOLUMES outside [1, 10]");       \
  } while (0)
#endif

#define PI_FLOAT 3.14159265358979323846f
#define FOUR_PI_INV_FLOAT                                                      \
  0.0795774715459476678844f // 1 / (4 \pi), from Wolfram Alpha

extern "C" {
__device__ static void calculate_solid_angle(
    float *world_from_index, // (3, 3) array giving the world_from_index ray
                             // transform for the camera
    float *solid_angle,      // flat array, with shape (out_height, out_width).
    int udx,                 // index into image width
    int vdx,                 // index into image height
    int img_dx               // index into solid_angle
) {
  /**
   * SOLID ANGLE CALCULATION
   *
   * Let the pixel's four corners be c0, c1, c2, c3.  Split the pixel into two
   * right triangles.  These triangles each form a tetrahedron with the X-ray
   * source S.  We can then use a solid-angle-of-tetrahedron formula.
   *
   * From Wikipedia:
   *      Let OABC be the vertices of a tetrahedron with an origin at O
   * subtended by the triangular face ABC where \vec{a}, \vec{b}, \vec{c} are
   * the vectors \vec{SA}, \vec{SB}, \vec{SC} respectively.  Then,
   *
   * tan(\Omega / 2) = NUMERATOR / DENOMINATOR, with
   *
   * NUMERATOR = \vec{a} \cdot (\vec{b} \times \vec{c})
   * DENOMINATOR = abc + (\vec{a} \cdot \vec{b}) c + (\vec{a} \cdot \vec{c}) b
   * +
   * (\vec{b} \cdot \vec{c}) a
   *
   * where a,b,c are the magnitudes of their respective vectors.
   *
   * There are two potential pitfalls with the above formula.
   * 1. The NUMERATOR (a scalar triple product) can be negative if \vec{a},
   * \vec{b}, \vec{c} have the wrong winding.  Since no other portion of the
   * formula depends on the winding, computing the absolute value of the
   * scalar triple product is sufficient.
   * 2. If the NUMERATOR is positive but the DENOMINATOR is negative, the
   * formula returns a negative value that must be increased by \pi.
   */

  /*
   * PIXEL DIAGRAM
   *
   * corner0 __ corner1
   *        |__|
   * corner3    corner2
   */
  float cx[4];   // source-to-corner vector x-values in world space
  float cy[4];   // source-to-corner vector y-values in world space
  float cz[4];   // source-to-corner vector z-values in world space
  float cmag[4]; // magnitude of source-to-corner vector

  float cu_offset[4] = {0.f, 1.f, 1.f, 0.f};
  float cv_offset[4] = {0.f, 0.f, 1.f, 1.f};
  for (int c = 0; c < 4; c++) {
    float cu = udx + cu_offset[c];
    float cv = vdx + cv_offset[c];

    cx[c] = cu * world_from_index[0] + cv * world_from_index[1] +
            world_from_index[2];
    cy[c] = cu * world_from_index[3] + cv * world_from_index[4] +
            world_from_index[5];
    cz[c] = cu * world_from_index[6] + cv * world_from_index[7] +
            world_from_index[8];

    cmag[c] = sqrtf((cx[c] * cx[c]) + (cy[c] * cy[c]) + (cz[c] * cz[c]));
  }

  /*
   * The cross- and dot-products needed for the [c0, c1, c2] triangle are:
   *
   * - absolute value of triple product of c0,c1,c2 = c1 \cdot (c0 \times c2)
   *      Since the magnitude of the triple product is invariant under
   * reorderings of the three vectors, we choose to cross-product c0,c2 so we
   * can reuse that result
   * - dot product of c0, c1
   * - dot product of c0, c2
   * - dot product of c1, c2
   *
   * The products needed for the [c0, c2, c3] triangle are:
   *
   * - absolute value of triple product of c0,c2,c3 = c3 \cdot (c0 \times c2)
   *      Since the magnitude of the triple product is invariant under
   * reorderings of the three vectors, we choose to cross-product c0,c2 so we
   * can reuse that result
   * - dot product of c0, c2
   * - dot product of c0, c3
   * - dot product of c2, c3
   *
   * Thus, the cross- and dot-products to compute are:
   *  - c0 \times c2
   *  - c0 \dot c1
   *  - c0 \dot c2
   *  - c0 \dot c3
   *  - c1 \dot c2
   *  - c2 \dot c3
   */
  float c0_cross_c2_x = (cy[0] * cz[2]) - (cz[0] * cy[2]);
  float c0_cross_c2_y = (cz[0] * cx[2]) - (cx[0] * cz[2]);
  float c0_cross_c2_z = (cx[0] * cy[2]) - (cy[0] * cx[2]);

  float c0_dot_c1 = (cx[0] * cx[1]) + (cy[0] * cy[1]) + (cz[0] * cz[1]);
  float c0_dot_c2 = (cx[0] * cx[2]) + (cy[0] * cy[2]) + (cz[0] * cz[2]);
  float c0_dot_c3 = (cx[0] * cx[3]) + (cy[0] * cy[3]) + (cz[0] * cz[3]);
  float c1_dot_c2 = (cx[1] * cx[2]) + (cy[1] * cy[2]) + (cz[1] * cz[2]);
  float c2_dot_c3 = (cx[2] * cx[3]) + (cy[2] * cy[3]) + (cz[2] * cz[3]);

  float numer_012 = fabs((cx[1] * c0_cross_c2_x) + (cy[1] * c0_cross_c2_y) +
                         (cz[1] * c0_cross_c2_z));
  float numer_023 = fabs((cx[3] * c0_cross_c2_x) + (cy[3] * c0_cross_c2_y) +
                         (cz[3] * c0_cross_c2_z));

  float denom_012 = (cmag[0] * cmag[1] * cmag[2]) + (c0_dot_c1 * cmag[2]) +
                    (c0_dot_c2 * cmag[1]) + (c1_dot_c2 * cmag[0]);
  float denom_023 = (cmag[0] * cmag[2] * cmag[3]) + (c0_dot_c2 * cmag[3]) +
                    (c0_dot_c3 * cmag[2]) + (c2_dot_c3 * cmag[0]);

  float solid_angle_012 = 2.f * atan2(numer_012, denom_012);
  if (solid_angle_012 < 0.0f) {
    solid_angle_012 += PI_FLOAT;
  }
  float solid_angle_023 = 2.f * atan2(numer_023, denom_023);
  if (solid_angle_023 < 0.0f) {
    solid_angle_023 += PI_FLOAT;
  }

  solid_angle[img_dx] = solid_angle_012 + solid_angle_023;
}

__global__ void projectKernel(
    int out_width,  // width of the output image
    int out_height, // height of the output image
    float step,     // step size (TODO: in world)
    int *priority,  // volumes with smaller priority-ID have higher priority
                    // when determining which volume we are in
    float *gVolumeEdgeMinPointX, // These give a bounding box in world-space
                                 // around each volume.
    float *gVolumeEdgeMinPointY, 
    float *gVolumeEdgeMinPointZ,
    float *gVolumeEdgeMaxPointX, 
    float *gVolumeEdgeMaxPointY,
    float *gVolumeEdgeMaxPointZ,
    float *gVoxelElementSizeX, // one value for each of the NUM_VOLUMES volumes
    float *gVoxelElementSizeY, 
    float *gVoxelElementSizeZ,
    float sx,      // x-coordinate of source point for rays in world-space
    float sy,      // y-coordinate of source point for rays in world-space
    float sz,      // z-coordinate of source point for rays in world-space
    float *sx_ijk, // x-coordinate of source point in IJK space for each
                   // volume (NUM_VOLUMES,)
    float *sy_ijk, // y-coordinate of source point in IJK space for each
                   // volume (NUM_VOLUMES,)
    float *sz_ijk, // z-coordinate of source point in IJK space for each
                   // volume (NUM_VOLUMES,) (passed in to avoid re-computing
                   // on every thread)
    float max_ray_length,    // max distance a ray can travel
    float *world_from_index, // (3, 3) array giving the world_from_index ray
                             // transform for the camera
    float *ijk_from_world, // (NUM_VOLUMES, 3, 4) transform giving the transform
                           // from world to IJK coordinates for each volume.
    int n_bins,            // the number of spectral bins
    float *energies,       // 1-D array -- size is the n_bins. Units: [keV]
    float *pdf, // 1-D array -- probability density function over the energies
    float *absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that
                              // represents the precomputed
                              // get_absorption_coef values. index into the
                              // table as: table[bin * NUM_MATERIALS + mat]
    float *intensity,         // flat array, with shape (out_height, out_width).
    float *photon_prob,       // flat array, with shape (out_height, out_width).
    float *solid_angle,       // flat array, with shape (out_height, out_width).
                              // Could be NULL pointer
    int offsetW, 
    int offsetH,
    cudaTextureObject_t volume_tex_0, //cudaTextureObject_t
    cudaTextureObject_t  seg_tex_0,
    cudaTextureObject_t  seg_tex_1,
    //cudaTextureObject_t  seg_tex_2,
    float *out
  ) {
    cudaTextureObject_t volume_tex[] = {volume_tex_0};
    cudaTextureObject_t seg_tex[] = {seg_tex_0, seg_tex_1};
  // The output image has the following coordinate system, with cell-centered
  // sampling. y is along the fast axis (columns), x along the slow (rows).
  //
  //      x -->
  //    y *---------------------------*
  //    | |                           |
  //    V |                           |
  //      |        output image       |
  //      |                           |
  //      |                           |
  //      *---------------------------*
  //
  //

  int udx = threadIdx.x + (blockIdx.x + offsetW) *
                              blockDim.x; // index into output image width
  int vdx = threadIdx.y + (blockIdx.y + offsetH) *
                              blockDim.y; // index into output image height
  // int debug = (udx == 973) && (vdx == 598); // larger image size
  // int debug = (udx == 243) && (vdx == 149); // 4x4 binning

  // if (udx == 40) printf("udx: %d, vdx: %d\n", udx, vdx);

  // if the current point is outside the output image, no computation needed
  if (udx >= out_width || vdx >= out_height)
    return;
  
  // flat index to pixel in *intensity and *photon_prob
  // int img_dx = vdx * out_width + udx;
  int img_dx = (udx * out_height) + vdx;

  // initialize intensity and photon_prob to 0
  intensity[img_dx] = 0.0f;
  photon_prob[img_dx] = 0.0f;

  if (NULL != solid_angle) {
    calculate_solid_angle(world_from_index, solid_angle, udx, vdx, img_dx);
  }

  // cell-centered sampling point corresponding to pixel index, in
  // index-space.
  float u = (float)udx + 0.5;
  float v = (float)vdx + 0.5;

  // Vector in world-space along ray from source-point to pixel at [u,v] on
  // the detector plane.
  float rx =
      u * world_from_index[0] + v * world_from_index[1] + world_from_index[2];
  float ry =
      u * world_from_index[3] + v * world_from_index[4] + world_from_index[5];
  float rz =
      u * world_from_index[6] + v * world_from_index[7] + world_from_index[8];

  /* make the ray a unit vector */
  float ray_length = sqrtf(rx * rx + ry * ry + rz * rz);
  float inv_ray_norm = 1.0f / ray_length;
  rx *= inv_ray_norm;
  ry *= inv_ray_norm;
  rz *= inv_ray_norm;

  // calculate projections
  // Part 1: compute alpha value at entry and exit point of all volumes on
  // either side of the ray, in world-space. minAlpha: the distance from
  // source point to all-volumes entry point of the ray, in world-space.
  // maxAlpha: the distance from source point to all-volumes exit point of the
  // ray.
  float minAlpha = INFINITY; // the furthest along the ray we want to consider
                             // is the start point.
  float maxAlpha = 0;        // closest point to consider is at the detector
  float minAlpha_vol[NUM_VOLUMES],
      maxAlpha_vol[NUM_VOLUMES]; // same, but just for each volume.
  float alpha0, alpha1, reci;
  int do_trace[NUM_VOLUMES]; // for each volume, whether or not to perform the
                             // ray-tracing
  int do_return = 1;

  // Get the ray direction in the IJK space for each volume.
  float rx_ijk[NUM_VOLUMES];
  float ry_ijk[NUM_VOLUMES];
  float rz_ijk[NUM_VOLUMES];
  int offs = 12; // TODO: fix bad style
  for (int i = 0; i < NUM_VOLUMES; i++) {
    // Homogeneous transform of a vector.
    rx_ijk[i] =
        ijk_from_world[offs * i + 0] * rx + ijk_from_world[offs * i + 1] * ry +
        ijk_from_world[offs * i + 2] * rz + ijk_from_world[offs * i + 3] * 0;
    ry_ijk[i] =
        ijk_from_world[offs * i + 4] * rx + ijk_from_world[offs * i + 5] * ry +
        ijk_from_world[offs * i + 6] * rz + ijk_from_world[offs * i + 7] * 0;
    rz_ijk[i] =
        ijk_from_world[offs * i + 8] * rx + ijk_from_world[offs * i + 9] * ry +
        ijk_from_world[offs * i + 10] * rz + ijk_from_world[offs * i + 11] * 0;

    // Get the number of times the ijk ray can fit between the source and the
    // entry/exit points of this volume in *this* IJK space.
    do_trace[i] = 1;
    minAlpha_vol[i] = 0;
    maxAlpha_vol[i] = max_ray_length > 0 ? max_ray_length : INFINITY;

    if (0.0f != rx_ijk[i]) {
      reci = 1.0f / rx_ijk[i];
      alpha0 = (gVolumeEdgeMinPointX[i] - sx_ijk[i]) * reci;
      alpha1 = (gVolumeEdgeMaxPointX[i] - sx_ijk[i]) * reci;



      minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
      maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
    } else if (gVolumeEdgeMinPointX[i] > sx_ijk[i] ||
               sx_ijk[i] > gVolumeEdgeMaxPointX[i]) {
      do_trace[i] = 0;
      continue;
    }
    if (0.0f != ry_ijk[i]) {
      reci = 1.0f / ry_ijk[i];
      alpha0 = (gVolumeEdgeMinPointY[i] - sy_ijk[i]) * reci;
      alpha1 = (gVolumeEdgeMaxPointY[i] - sy_ijk[i]) * reci;
      minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
      maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
    } else if (gVolumeEdgeMinPointY[i] > sy_ijk[i] ||
               sy_ijk[i] > gVolumeEdgeMaxPointY[i]) {
      do_trace[i] = 0;
      continue;
    }
    if (0.0f != rz_ijk[i]) {
      reci = 1.0f / rz_ijk[i];
      alpha0 = (gVolumeEdgeMinPointZ[i] - sz_ijk[i]) * reci;
      alpha1 = (gVolumeEdgeMaxPointZ[i] - sz_ijk[i]) * reci;
      minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
      maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
    } else if (gVolumeEdgeMinPointZ[i] > sz_ijk[i] ||
               sz_ijk[i] > gVolumeEdgeMaxPointZ[i]) {
      do_trace[i] = 0;
      continue;
    }
    do_return = 0;

    // Now, this is valid, since "how many times the ray can fit in the
    // distance" is equivalent to the distance in world space, since [rx, ry,
    // rz] is a unit vector.
    minAlpha = fmin(minAlpha, minAlpha_vol[i]);
    maxAlpha = fmax(maxAlpha, maxAlpha_vol[i]);
  }

  // Means none of the volumes have do_trace = 1.
  if (do_return)
    return;

  // printf("global min, max alphas: %f, %f\n", minAlpha, maxAlpha);

  // Part 2: Cast ray if it intersects any of the volumes
  int num_steps = ceil((maxAlpha - minAlpha) / step);
  // if (debug) printf("num_steps: %d\n", num_steps);

  // initialize the projection-output to 0.
  float area_density[NUM_MATERIALS];
  for (int m = 0; m < NUM_MATERIALS; m++) {
    area_density[m] = 0.0f;
  }


  float area_density2[NUM_MATERIALS];
  for (int m = 0; m < NUM_MATERIALS; m++) {
    area_density2[m] = 0.0f;
  }
  float px[NUM_VOLUMES]; // voxel-space point
  float py[NUM_VOLUMES];
  float pz[NUM_VOLUMES];
  float alpha = minAlpha;      // distance along the world space ray (alpha =
                               // minAlpha[i] + step * t)
  int curr_priority;           // the priority at the location
  int n_vols_at_curr_priority; // how many volumes to consider at the location
  float seg_at_alpha[NUM_VOLUMES][NUM_MATERIALS];
  // if (debug) printf("start trace\n");

  // Attenuate up to minAlpha, assuming it is filled with air.
  if (ATTENUATE_OUTSIDE_VOLUME) {
    area_density[AIR_INDEX] += (minAlpha / step) * AIR_DENSITY;
  }

  // trace (if doing the last segment separately, need to use num_steps - 1
  for (int t = 0; t < num_steps; t++) {
    
    //printf("test");
     
    // do { 
      
    //   if (do_trace[0]) { 
    //     do { 
    //       px[0] = sx_ijk[0] + alpha * rx_ijk[0] - 0.5; 
    //       py[0] = sy_ijk[0] + alpha * ry_ijk[0] - 0.5; 
    //       pz[0] = sz_ijk[0] + alpha * rz_ijk[0] - 0.5; } 
    //       while (0); 
          
    //     do { 
    //       do { 
            
    //         float value = round( cubicTex3D(seg_tex[0 * 3 + 0], make_float3(px[0], py[0], pz[0]))); 
    //         seg_at_alpha[0][0] = value;
    //         } 
    //         while (0);
            
    //       do { 
    //         float value = round( cubicTex3D(seg_tex[0 * 3 + 1], make_float3(px[0], py[0], pz[0]))); 
    //         seg_at_alpha[0][1] = value; 
    //         } 
    //         while (0);
    //       do { 
    //         float value = round( cubicTex3D(seg_tex[0 * 3 + 2], make_float3(px[0], py[0], pz[0]))); 
    //         seg_at_alpha[0][2] = value; 
    //         } 
    //         while (0); } 
    //       while (0); 
    //     } 
    //   } 
    //   while (0); 
        
    LOAD_SEGS_AT_ALPHA; // initializes p{x,y,z}[...] and
                        // seg_at_alpha[...][...]
    // if (debug) printf("  loaded segs\n"); // This is the one that seems to
    // take a half a second.
    //

    curr_priority = NUM_VOLUMES;
    n_vols_at_curr_priority = 0;
    for (int i = 0; i < NUM_VOLUMES; i++) {
      if (0 == do_trace[i]) {
        continue;
      }
      if ((alpha < minAlpha_vol[i]) || (alpha > maxAlpha_vol[i])) {
        continue;
      }
      float any_seg = 0.0f;
      for (int m = 0; m < NUM_MATERIALS; m++) {
        any_seg += seg_at_alpha[i][m];
        if (any_seg > 0.0f) {
          break;
        }
      }
      if (0.0f == any_seg) {
        continue;
      }

      if (priority[i] < curr_priority) {
        curr_priority = priority[i];
        n_vols_at_curr_priority = 1;
      } else if (priority[i] == curr_priority) {
        n_vols_at_curr_priority += 1;
      }
    }

 

    // if (debug) printf("  got priority at alpha, num vols\n"); // This is
    // the one that seems to take a half a second.
    if (0 == n_vols_at_curr_priority) {
      // Outside the bounds of all volumes to trace. Use the default
      // AIR_DENSITY.
      if (ATTENUATE_OUTSIDE_VOLUME) {
        area_density[AIR_INDEX] += AIR_DENSITY;
      }
    } else {
      float weight = 1.0f / ((float)n_vols_at_curr_priority);

      // For the entry boundary, multiply by 0.5. That is, for the initial
      // interpolated value, only a half step-size is considered in the
      // computation. For the second-to-last interpolation point, also
      // multiply by 0.5, since there will be a final step at the
      // globalMaxAlpha boundary.
      weight *= (0 == t || num_steps - 1 == t) ? 0.5f : 1.0f;

      do { 
        if (do_trace[0] && (priority[0] == curr_priority)) { 
          do { 
            do { 
              area_density[(0)] += (weight)*tex3D<float>(volume_tex[0], px[0], py[0], pz[0]) * seg_at_alpha[0][0]; } 
            while (0); 
            do { 
              area_density[(1)] += (weight)*tex3D<float>(volume_tex[0], px[0], py[0], pz[0]) * seg_at_alpha[0][1]; } 
            while (0); 
            do { 
              area_density[(2)] += (weight)*tex3D<float>(volume_tex[0], px[0], py[0], pz[0]) * seg_at_alpha[0][2]; } 
              while (0); 
            }
            while (0);
          } 
        } while (0);
      // INTERPOLATE(weight);
      
      // for (int m = 0; m < NUM_MATERIALS; m++) {
      //   out[m] = area_density[m];
      // }
      

    }
    alpha += step;
  }
  
  // Attenuate from the end of the volume to the detector.
  if (ATTENUATE_OUTSIDE_VOLUME) {
    area_density[AIR_INDEX] += (ray_length - maxAlpha) / step * AIR_DENSITY;
  }

  // if (debug) printf("finished trace, num_steps: %d\n", num_steps);

  // Scaling by step
  for (int m = 0; m < NUM_MATERIALS; m++) {
    area_density[m] *= step;
  }

  // Convert to centimeters
  for (int m = 0; m < NUM_MATERIALS; m++) {
    area_density[m] /= 10.0f;
  }

  /* Up to this point, we have accomplished the original projectKernel
   * functionality. The next steps to do are combining the forward_projections
   * dictionary-ization and the mass_attenuation computation
   */

  // forward_projections dictionary-ization is implicit.

  // MASS ATTENUATION COMPUTATION

  /**
   * EXPLANATION OF THE PHYSICS/MATHEMATICS
   *
   *      The mass attenuation coefficient (found in absorb_coef_table) is:
   * \mu / \rho, where \mu is the linear attenuation coefficient, and \rho is
   * the mass density.  \mu has units of inverse length, and \rho has units of
   * mass/volume, so the mass attenuation coefficient has units of [cm^2 / g]
   *      area_density[m] is the product of [linear distance of the ray
   * through material 'm'] and [density of the material].  Accordingly,
   * area_density[m] has units of [g / cm^2].
   *
   * The mass attenuation code uses the Beer-Lambert law:
   *
   *      I = I_{0} exp[-(\mu / \rho) * \rho * d]
   *
   * where I_{0} is the initial intensity, (\mu / \rho) is the mass
   * attenuation coefficient, \rho is the density, and d is the length of the
   * ray passing through the material.  Note that the product (\rho * d), also
   * known as the 'area density' is the quantity area_density[m]. Because we
   * are attenuating multiple materials, the exponent that we use for the
   * Beer-Lambert law is the sum of the (\mu_{mat} / \rho_{mat}) * (\rho_{mat}
   * * d_{mat}) for each material 'mat'.
   *
   *      The above explains the calculation up to and including
   *              '____ = expf(-1 * beer_lambert_exp)',
   * but does not yet explain the remaining calculation.  The remaining
   * calculation serves to approximate the workings of a pixel in the
   * dectector:
   *
   *      pixelReading = \sum_{E} attenuatedBeamStrength[E] * E * p(E)
   *
   * where attenuatedBeamStrength follows the Beer-Lambert law as above, E is
   * the energies of the spectrum, and p(E) is the PDF of the spectrum. Note
   * also that the Beer-Lambert law deals with the quantity 'intensity', which
   * is related to the power transmitted through [unit area perpendicular to
   * the direction of travel]. Since the intensities mentioned in the
   * Beer-Lambert law are proportional to 1/[unit area], we can replace the
   * "intensity" calcuation with simply the energies involved.  Later
   * conversion to other physical quanities can be done outside of the kernel.
   */
  // if (debug)  printf("attenuation\n");

  for (int bin = 0; bin < n_bins; bin++) {
    float beer_lambert_exp = 0.0f;
    for (int m = 0; m < NUM_MATERIALS; m++) {
      beer_lambert_exp +=
          area_density[m] * absorb_coef_table[bin * NUM_MATERIALS + m];
    }
    float photon_prob_tmp =
        expf(-1.f * beer_lambert_exp) * pdf[bin]; // dimensionless value

    photon_prob[img_dx] += photon_prob_tmp;
    intensity[img_dx] +=
        energies[bin] *
        photon_prob_tmp; // units: [keV] per unit photon to hit the pixel
  }

  // if (debug) printf("done with kernel thread\n");
  return;
}

/*** KERNEL RESAMPLING FUNCTION ***/
/**
 * It's placed here so that it can properly access the CUDA textures of the
 * volumes and segmentations
 */

#if NUM_MATERIALS == 1
#define RESAMPLE_TEXTURES(vol_id)                                              \
  do {                                                                         \
    density_sample[vol_id] = tex3D<float>(VOL_TEX(vol_id),inp_x, inp_y, inp_z);       \
    mat_sample[vol_id][0] = cubicTex3D(SEG_TEX(vol_id,0), make_float3(inp_x, inp_y, inp_z));   \
  } while (0)
#elif NUM_MATERIALS == 2
#define RESAMPLE_TEXTURES(vol_id)                                              \
  do {                                                                         \
    density_sample[vol_id] = tex3D<float>(VOL_TEX(vol_id), inp_x, inp_y, inp_z);       \
    mat_sample[vol_id][0] = cubicTex3D(SEG_TEX(vol_id,0), make_float3(inp_x, inp_y, inp_z));   \
    mat_sample[vol_id][1] = cubicTex3D(SEG_TEX(vol_id,1), make_float3(inp_x, inp_y, inp_z));   \
  } while (0)
#elif NUM_MATERIALS == 3
#define RESAMPLE_TEXTURES(vol_id)                                              \
    do {                                                                         \
      density_sample[vol_id] = tex3D<float>(VOL_TEX(vol_id), inp_x, inp_y, inp_z);       \
      mat_sample[vol_id][0] = cubicTex3D(SEG_TEX(vol_id,0), make_float3(inp_x, inp_y, inp_z));   \
      mat_sample[vol_id][1] = cubicTex3D(SEG_TEX(vol_id,1), make_float3(inp_x, inp_y, inp_z));   \
      mat_sample[vol_id][2] = cubicTex3D(SEG_TEX(vol_id,2), make_float3(inp_x, inp_y, inp_z));   \
    } while (0)
#else
#define RESAMPLE_TEXTURES(vol_id)                                              \
  do {                                                                         \
    printf("NUM_MATERIALS not in [1, 14]");                                    \
  } while (0)
#endif

__global__ void resample_megavolume(
    int *inp_priority,
    int *inp_voxelBoundX, // number of voxels in x direction for each volume
    int *inp_voxelBoundY, int *inp_voxelBoundZ,
    float *inp_ijk_from_world, // ijk_from_world transforms for input volumes
                               // TODO: is each transform 3x3?
    float megaMinX, // bounding box for output megavolume, in world coordinates
    float megaMinY, float megaMinZ, float megaMaxX, float megaMaxY,
    float megaMaxZ,
    float megaVoxelSizeX, // voxel size for output megavolume, in world
                          // coordinates
    float megaVoxelSizeY, float megaVoxelSizeZ,
    int mega_x_len, // the (exclusive, upper) array index bound of the
                    // megavolume
    int mega_y_len, int mega_z_len,
    float *output_density, // volume-sized array
    char *output_mat_id,   // volume-sized array to hold the material IDs of the
                           // voxels,
    int offsetX, int offsetY, int offsetZ,
    const unsigned long long volume_tex[NUM_VOLUMES],
    const unsigned long long seg_tex[NUM_VOLUMES*NUM_MATERIALS]
    ) {
  /*
   * Sample in voxel centers.
   *
   * Loop keeps track of {x,y,z} position in world coord.s as well as IJK
   * indices for megavolume voxels. The first voxel has IJK indices (0,0,0)
   * and is centered at (minX + 0.5 * voxX, minY + 0.5 * voxY, minZ + 0.5 *
   * voxZ)
   *
   * The upper bound of the loop checking for:
   *       {x,y,z} <= megaMax{X,Y,Z}
   * is sufficient because the preprocessing of the boudning box ensured that
   * the voxels fit neatly into the bounding box
   */

  // local storage to store the results of the tex3D calls.
  // As a switch, we rely on the fact that the results of the tex3D calls
  // should never be negative
  float density_sample[NUM_VOLUMES];
  // local storage to store the results of the cubicTex3D calls
  float mat_sample[NUM_VOLUMES][NUM_MATERIALS];

  int x_low = threadIdx.x + (blockIdx.x + offsetX) *
                                blockDim.x; // the x-index of the lowest voxel
  int y_low = threadIdx.y + (blockIdx.y + offsetY) * blockDim.y;
  int z_low = threadIdx.z + (blockIdx.z + offsetZ) * blockDim.z;

  int x_high = min(x_low + blockDim.x, mega_x_len);
  int y_high = min(y_low + blockDim.y, mega_y_len);
  int z_high = min(z_low + blockDim.z, mega_z_len);

  if ((x_low == 0) && (y_low == 0) && (z_low == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0) && (threadIdx.z == 0)) {
    printf("blockDim: {%d, %d, %d}\n", blockDim.x, blockDim.y, blockDim.z);
  }

  for (int x_ind = x_low; x_ind < x_high; x_ind++) {
    for (int y_ind = y_low; y_ind < y_high; y_ind++) {
      for (int z_ind = z_low; z_ind < z_high; z_ind++) {
        float x = megaMinX + (0.5f + (float)x_ind) * megaVoxelSizeX;
        float y = megaMinY + (0.5f + (float)y_ind) * megaVoxelSizeY;
        float z = megaMinZ + (0.5f + (float)z_ind) * megaVoxelSizeZ;
        // for each volume, check whether we are inside its bounds
        int curr_priority = NUM_VOLUMES;

        for (int i = 0; i < NUM_VOLUMES; i++) {
          density_sample[i] = -1.0f; // "reset" this volume's sample

          int offset = 3 * 4 * i; // TODO: check that this matrix
                                  // multiplication is done properly
          float inp_x = (inp_ijk_from_world[offset + 0] * x) +
                        (inp_ijk_from_world[offset + 1] * y) +
                        (inp_ijk_from_world[offset + 2] * z) +
                        inp_ijk_from_world[offset + 3];
          if ((inp_x < 0.0) || (inp_x >= inp_voxelBoundX[i]))
            continue; // TODO: make sure this behavior agrees with the
                      // behavior of ijk_from_world transforms

          float inp_y = (inp_ijk_from_world[offset + 4] * x) +
                        (inp_ijk_from_world[offset + 5] * y) +
                        (inp_ijk_from_world[offset + 6] * z) +
                        inp_ijk_from_world[offset + 7];
          if ((inp_y < 0.0) || (inp_y >= inp_voxelBoundY[i]))
            continue;

          float inp_z = (inp_ijk_from_world[offset + 8] * x) +
                        (inp_ijk_from_world[offset + 9] * y) +
                        (inp_ijk_from_world[offset + 10] * z) +
                        inp_ijk_from_world[offset + 11];
          if ((inp_z < 0.0) || (inp_z >= inp_voxelBoundZ[i]))
            continue;

          if (inp_priority[i] < curr_priority)
            curr_priority = inp_priority[i];
          else if (inp_priority[i] > curr_priority)
            continue;

          // mjudish understands that this is ugly, but it compiles
          if (0 == i) {
            RESAMPLE_TEXTURES(0);
          }
#if NUM_VOLUMES > 1
          else if (1 == i) {
            RESAMPLE_TEXTURES(1);
          }
#endif
#if NUM_VOLUMES > 2
          else if (2 == i) {
            RESAMPLE_TEXTURES(2);
          }
#endif
#if NUM_VOLUMES > 20
#define INTERPOLATE(multiplier)                                                \
  do {                                                                         \
    printf("INTERPOLATE not supported for NUM_VOLUMES outside [1, 10]");       \
  } while (0)
#endif
          // Maximum supported value of NUM_VOLUMES is 10
        }

        int output_idx =
            x_ind + (y_ind * mega_x_len) + (z_ind * mega_x_len * mega_y_len);
        if (NUM_VOLUMES == curr_priority) {
          // no input volumes at the current point
          output_density[output_idx] = 0.0f;
          output_mat_id[output_idx] = NUM_MATERIALS; // out of range for mat id,
                                                     // so indicates no material
        } else {
          // for averaging the densities of the volumes to "mix"
          int n_vols_at_curr_priority = 0;
          float total_density = 0.0f;

          // for determining the material most
          float total_mat_seg[NUM_MATERIALS];
          for (int m = 0; m < NUM_MATERIALS; m++) {
            total_mat_seg[m] = 0.0f;
          }

          for (int i = 0; i < NUM_VOLUMES; i++) {
            if (curr_priority == inp_priority[i]) {
              n_vols_at_curr_priority++;
              total_density += density_sample[i];

              for (int m = 0; m < NUM_MATERIALS; m++) {
                total_mat_seg[m] = mat_sample[i][m];
              }
            }
          }

          int mat_id = NUM_MATERIALS;
          float highest_mat_seg = 0.0f;
          for (int m = 0; m < NUM_MATERIALS; m++) {
            if (total_mat_seg[m] > highest_mat_seg) {
              mat_id = m;
              highest_mat_seg = total_mat_seg[m];
            }
          }

          output_density[output_idx] =
              total_density / ((float)n_vols_at_curr_priority);
          output_mat_id[output_idx] = mat_id;
        }
      }
    }
  }

  return;
}
}
