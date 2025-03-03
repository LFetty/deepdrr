/*
 * This file contains the declarations of the CUDA textures for:
 *  - NUM_VOLUMES CT volumes
 *  - (NUM_MATERIALS * NUM_VOLUMES) segmentation channels
 */

#define SEG_PASTER(vol_id, mat_id) seg_##vol_id##_##mat_id
#define SEG(vol_id, mat_id) SEG_PASTER(vol_id, mat_id)
#define VOL_PASTER(vol_id) volume_##vol_id
#define VOLUME(vol_id) VOL_PASTER(vol_id)

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
#if NUM_VOLUMES > 0
#define CURR_VOL_ID 0
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif

#undef CURR_VOL_ID
#endif
