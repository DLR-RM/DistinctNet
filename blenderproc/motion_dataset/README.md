# Motion Dataset Generation with BlenderProc

This folder contains config and an additional module for generating the synthetic training dataset for moving foreground segmentation.
It requires the SunCG dataset, but you can replace the respective Loader in the config if you want to use another synthetic dataset.

**Note:** BlenderProc builds its own python environment, so don't run this with the activated _distinctnet_ environment!

1. Download [BlenderProc](https://github.com/DLR-RM/BlenderProc).

2. Place the [ObjectPoseShifter](./ObjectPoseShifter.py) module in `<BlenderProcRoot>/src/object/`.

3. Inside the BlenderProc root, run `python run.py <DistinctNetRoot>/blenderproc/motion_dataset/config.yaml </path/to/house.json> </path/to/output_dir>`.

This generates 10 hdf5 files in `/path/to/output_dir`. Each file contains

- RGB image
- Segmentation map of the object that is in motion across the 10 files

4. (optional) Run [check.py](./../check.py) to visualize images.

Running this *n* times creates *n* sequences (paper: 4,500 train / 500 val sequences, totalling in 50,000 images).
See also [the rerun option](https://github.com/DLR-RM/BlenderProc#general).
Please see header of the [motion dataset class](./../../data/motion_dataset.py) for more details on assumed folder structure.
For more details on BlenderProc we refer to [its official documentation](https://dlr-rm.github.io/BlenderProc/).
