import bpy
from collections import defaultdict
import mathutils

import random
from src.main.Module import Module
from src.utility.BlenderUtility import check_intersection, check_bb_intersection, get_all_blender_mesh_objects
from mathutils import Vector


class ObjectPoseShifter(Module):
    """
        Shifts the 6D pose of a random visible object.
        Example 1: Shift a random object by

        {
          "module": "object.ObjectPoseShifter",
          "config": {
            "x_shift": 0.25,
            "y_shift": 0.25,
            "z_shift": 0.25,
            "rx_shift": 0.4,
            "ry_shift": 0.4,
            "rz_shift": 0.4,
          }
        },

    .. csv-table::
        :header: "Parameter", "Description"

        "{x,y,z}_shift", "Maximum distance shifting [m]"
        "{rx,ry,rz}_shift", "Maximum rotation change [rad]"
    """

    def __init__(self, config):
        Module.__init__(self, config)
        self.sqrt_number_of_rays = config.get_int("sqrt_number_of_rays", 10)


    def run(self):
        """ Shifts desired objects, and inserts a new key frame.
        """

        # obtain objects from visible objects
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data
        cam2world_matrix = cam_ob.matrix_world
        valid_objects = self._visible_objects(cam, cam2world_matrix)

        random.shuffle(valid_objects)
        obj = random.sample(valid_objects, 1)[0]
        self._add_seg_properties(obj)
        start = bpy.context.scene.frame_start
        end = bpy.context.scene.frame_end
        init_loc = obj.location.copy()
        init_rot = obj.rotation_euler.copy()
        loc = obj.location
        rot = obj.rotation_euler
        for frame in range(start, end):
            for i in range(0, 10):
                # sample here
                x_shift = self.config.get_float('x_shift', 0.1)
                y_shift = self.config.get_float('y_shift', 0.1)
                z_shift = self.config.get_float('z_shift', 0.1)
                rx_shift = self.config.get_float('rx_shift', 0.4)
                ry_shift = self.config.get_float('ry_shift', 0.4)
                rz_shift = self.config.get_float('rz_shift', 0.4)
                if i > 0:
                    num_tries = 0

                    # use 10 tries for sampling, otherwise skip
                    while num_tries < 10:
                        new_loc = Vector([init_loc.x + random.uniform(-x_shift, x_shift),
                                          init_loc.y + random.uniform(-y_shift, y_shift),
                                          init_loc.z + random.uniform(-z_shift, z_shift)])
                        new_rot = Vector([init_rot[0] + random.uniform(-rx_shift, rx_shift),
                                          init_rot[1] + random.uniform(-ry_shift, ry_shift),
                                          init_rot[2] + random.uniform(-rz_shift, rz_shift)])
                        obj.location = new_loc
                        obj.rotation_euler = new_rot
                        if self._is_still_visible(obj, cam, cam2world_matrix):
                            print('Took', num_tries, 'tries for sampling a pose shift')
                            break
                        else:
                            num_tries += 1
                            if num_tries >= 10:
                                print('Object is not visible after 10 tries. Exiting ...')
                                exit(0)
                else:
                    obj.location = init_loc
                    obj.rotation_euler = init_rot

                bpy.context.scene.frame_end += 1
                self._insert_key_frames(obj, frame_id=frame + i)
        bpy.context.scene.frame_end -= 1

    def _insert_key_frames(self, obj, frame_id):
        """ Inserts a new keyframe with the shifted object.

        :parameter obj: Shifted object
        :parameter frame_id: Frame id
        """
        obj.keyframe_insert(data_path='location', frame=frame_id)
        obj.keyframe_insert(data_path='rotation_euler', frame=frame_id)

    def _add_seg_properties(self, obj):
        """ Sets all custom properties of all given objects according to the configuration.

        :parameter obj: Selected object which should receive the custom properties
        """

        properties = {"cp_category_id": 1}

        for key, value in properties.items():
            if key.startswith("cp_"):
                key = key[3:]
                obj[key] = value
            else:
                raise RuntimeError(
                    "Loader modules support setting only custom properties. Use 'cp_' prefix for keys. "
                    "Use manipulators.Entity for setting object's attribute values.")

    def _is_still_visible(self, obj, cam, cam2world_matrix):
        """ Checks if the object is still visible after shifting.

        :param cam: The camera whose view frame is used (only FOV is relevant, pose of cam is ignored).
        :param cam2world_matrix: The world matrix which describes the camera orientation to check.
        :return: Boolean indicating if the object is still visible or not.
        """
        vis_objs = self._visible_objects(cam, cam2world_matrix)
        if obj in vis_objs:
            return True
        else:
            return False

    def _visible_objects(self, cam: bpy.types.Camera, cam2world_matrix: mathutils.Matrix):
        # adapted from src/camera/CameraSampler.py
        visible_objects = set()

        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        # Go in discrete grid-like steps over plane
        position = cam2world_matrix.to_translation()
        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                # Compute current point on plane
                end = frame[0] + vec_x * x / float(self.sqrt_number_of_rays - 1) + vec_y * y / float(
                    self.sqrt_number_of_rays - 1)
                # Send ray from the camera position through the current point on the plane
                _, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, position,
                                                                       end - position)
                # Add hit object to set
                visible_objects.add(hit_object)
        if len(visible_objects) == 0:
            print(f"Didn't find any objects. Exiting ...")
            exit(0)
        return list(visible_objects)
