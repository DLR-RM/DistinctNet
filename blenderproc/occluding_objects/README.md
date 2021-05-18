# Generate occluding objects with BlenderProc

This folder contains the config for generating occluding objects with BlenderProc.
These objects are later pasted onto the segmented manipulators to generate self-supervised manipulator-object images with corresponding annotations.
It requires the ShapeNet dataset, but you can replace the respective Loader in the config if you want to use another synthetic dataset.

**Note:** BlenderProc builds its own python environment, so don't run this with the activated _distinctnet_ environment!

1. Download [BlenderProc](https://github.com/DLR-RM/BlenderProc).

2. Inside the BlenderProc root, run `python run.py <DistinctNetRoot>/blenderproc/occluding_objects/config.yaml </path/to/ShapeNetCore> <synset_id> </path/to/output_dir>`.

This generates 5 hdf5 files in `/path/to/output_dir`. Each file contains

- RGB image of one ShapeNet object
- Segmentation map of the respective object

3. (optional) Run [check.py](./../check.py) to visualize images.

This should be run n times to create n sequences (paper: 4,500 train / 500 val sequences, totalling in 50,000 images).
Please see header of the [pasting script](../../gen/generate_training_data.py) for more details on assumed folder structure.
For more details on BlenderProc we refer to [its official documentation](https://dlr-rm.github.io/BlenderProc/).
