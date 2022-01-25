import os
import pymesh
import numpy as np
import itertools
import warnings

import Meshing.ElectrodeOperations as ElecOps
import FileOps.FileOperations as FileOps

class MeshOperations(ElecOps.ElectrodeOperations, FileOps.FileOperations):
    def __init__(self, surface_mesh: pymesh.Mesh, electrode_attributes: dict):
        self.merged_meshes: pymesh.Mesh = None
        self.skin_with_electrodes: list = None
        self.electrode_mesh: pymesh.Mesh = None
        self.surface_meshes: list = []
        ElecOps.ElectrodeOperations.__init__(self, surface_mesh, electrode_attributes)

    def phm_model_meshing(self, mesh_filename: str, resolve_intersections: bool=True, electrodes_to_omit: list=None) -> None:
        """Generate a .poly file of the provided model mesh.

        Args:
            mesh_filename (str): The path where to save the meshed model file
            resolve_intersections (bool, optional): Whether to resolve any self-intersections the model might have. Defaults to True.

        Raises:
            AttributeError: If the class variable `surface_meshes` does not contain the model meshes, throw the error.
        """
        if not self.surface_meshes:
            raise AttributeError('Meshes must be loaded first. Please load the meshes.')
        # self.electrode_meshing(electrodes_to_omit=electrodes_to_omit)

        # self.merged_meshes = pymesh.merge_meshes((self.skin_with_electrodes, *self.surface_meshes[1:]))
        self.merged_meshes = pymesh.merge_meshes((*self.surface_meshes,))
        regions = self.region_points(self.surface_meshes, 0.1, electrode_mesh=None)

        self_intersection = pymesh.detect_self_intersection(self.merged_meshes)
        if self_intersection.size != 0:
            if resolve_intersections:
                warnings.warn('Self intersections were detected. Trying to resolve', UserWarning)
            else:
                raise Warning('Self intersections were detected. User selected not to resolve.')

            temp_meshes = []
            combination_pairs = 2
            surface_meshes = np.array(self.surface_meshes)
            # surface_meshes[0] = self.skin_with_electrodes

            unique_intersections = np.unique(self.merged_meshes.get_attribute('face_sources')[self_intersection]).astype(np.int32)
            intersection_combinations = np.array(list(itertools.combinations(unique_intersections, combination_pairs))).astype(np.int32)

            for combination in intersection_combinations:
                if pymesh.detect_self_intersection(pymesh.merge_meshes((*surface_meshes[combination], ))).size != 0:
                    temp_meshes.append(pymesh.resolve_self_intersection(pymesh.merge_meshes((*surface_meshes[combination], ))))

            mask_intersecting = np.ones(len(surface_meshes), np.bool)
            mask_intersecting[unique_intersections] = False

            temp_meshes = temp_meshes + [surface_meshes[j] for j, i in enumerate(mask_intersecting) if mask_intersecting[j]]
            temp_merged_meshes = pymesh.merge_meshes((*temp_meshes, ))

            self.poly_write(mesh_filename, temp_merged_meshes.vertices, temp_merged_meshes.faces, regions)
        else:
            self.poly_write(mesh_filename, self.merged_meshes.vertices, self.merged_meshes.faces, regions)

    def sphere_model_meshing(self, mesh_filename: str) -> None:
        if not self.surface_meshes:
            raise AttributeError('Meshes must be loaded first. Please load the meshes.')
        self.electrode_meshing(sphere=True)

        self.merged_meshes = pymesh.merge_meshes((self.skin_with_electrodes, *self.surface_meshes[1:]))
        regions = self.region_points(self.surface_meshes, 0.1, electrode_mesh=self.electrode_mesh)

        self.poly_write(mesh_filename, self.merged_meshes.vertices, self.merged_meshes.faces, regions)

    def load_surface_meshes(self, base_path: str, file_names: list) -> None:
        # TODO: Order matters
        for file_name in file_names:
            self.surface_meshes.append(pymesh.load_mesh(os.path.join(base_path, file_name)))

    def electrode_meshing(self, sphere: bool=False, electrodes_to_omit: list=None):
        print("Placing Electrodes") # INFO log
        if sphere:
            self.sphere_electrode_positioning()
        else:
            self.standard_electrode_positioning(electrodes_to_omit)
        self.skin_with_electrodes = self.add_electrodes_on_skin()[0]

        electrodes_single_mesh = self.get_electrode_single_mesh()
        self.electrode_mesh = pymesh.boolean(electrodes_single_mesh, self._surface_mesh, 'difference')
        pymesh.map_face_attribute(electrodes_single_mesh, self.electrode_mesh, 'face_sources')

        return self.electrode_mesh

    @staticmethod
    def region_points(boundary_surfaces: list, shift: float, electrode_mesh: pymesh.Mesh=None, max_volume=30):
        points = {}
        for index, surface in enumerate(boundary_surfaces):
            vertices = surface.vertices
            max_z_point_id = np.where(vertices[:, 2] == np.amax(vertices[:, 2]))[0][0]
            max_z_point_unit_vector = vertices[max_z_point_id]/np.linalg.norm(vertices[max_z_point_id])

            points[str(index + 1)] = {
                'coordinates': np.round(vertices[max_z_point_id] - np.absolute(np.multiply(max_z_point_unit_vector, [0, 0, shift])), 6),
                'max_volume': max_volume,
            }

        if electrode_mesh is not None:
            electrode_regions = electrode_mesh.get_attribute('face_sources')
            electrode_mesh.add_attribute('vertex_valance')
            vertex_valance = electrode_mesh.get_attribute('vertex_valance')

            for region in np.unique(electrode_regions):
                faces = electrode_mesh.faces[np.where(electrode_regions == region)[0]]
                vertices = electrode_mesh.vertices[np.unique(faces)]

                max_valance_point = np.where(vertex_valance[np.unique(faces)] == np.amax(vertex_valance[np.unique(faces)]))[0][0]
                max_valance_unit_vector = vertices[max_valance_point]/np.linalg.norm(vertices[max_valance_point])

                points[str(region)] = {
                    'coordinates': np.round(vertices[max_valance_point] - np.multiply(max_valance_unit_vector, shift), 6),
                    'max_volume': max_volume,
                }

        return points
