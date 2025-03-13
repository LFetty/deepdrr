#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
import time

def main():
    output_dir = test_utils.get_output_dir()
    data_dir = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(
        data_dir
    data_dir = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(
        data_dir
    )
    #patient.faceup()
    #patient.faceup()

    # define the simulated C-arm
    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, alpha=90, beta=90, degrees=True, pixel_size=0.5)
    start_time = time.time()
    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, alpha=90, beta=90, degrees=True, pixel_size=0.5)
    start_time = time.time()
    # project in the AP view
    with deepdrr.Projector(
        volume=volume,
        carm=carm,
        step=0.1,  # stepsize along projection ray, measured in voxels
        spectrum="90KV_AL40", # energy spectrum
        photon_count=100000, # number of photons to simulate
        scatter_num=0, # number of scatter events to simulate
        neglog=True, # apply negative log transform to image (convenient for visualization)
        intensity_upper_bound=3, # Good default for windowing
    ) as projector:
        #carm.move_to(isocenter_in_world=volume.center_in_world + geo.v(0, 0, z))
        #print(f"Projecting at z={z}")
        image = projector.project()
    print(time.time()-start_time, 'sec')
    path = output_dir / "example_projector_new.png"
    image_utils.save(path, image)
    print(f"saved example projection image to {path.absolute()}")
# deepdrr with pycuda
# func:'initialize' args:[(<deepdrr.projector.projector.Projector object at 0x7f2c08be9250>,), {}] took: 0.3571 sec
# func:'project' args:[(<deepdrr.projector.projector.Projector object at 0x7f2c08be9250>,), {}] took: 1.1203 sec

# deepdrr with cupy and textures
# textures: 3x took: 0.0371-0.0560  sec
#           1x took: 0.1396-0.2484  sec
# func:'initialize' args:[(<deepdrr.projector.projector.Projector object at 0x7fc3bf72b7a0>,), {}] took: 0.5605-0.7536 sec
# func:'project' args:[(<deepdrr.projector.projector.Projector object at 0x7fc3bf72b7a0>,), {}] took: 1.1741-1.1348 sec


if __name__ == "__main__":
    main()
