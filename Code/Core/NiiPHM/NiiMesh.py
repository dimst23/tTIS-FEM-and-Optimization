from typing import Union, Tuple
import os
import PVGeo
import numpy as np
import pandas as pd
import pyvista as pv
import nibabel as nib

class NiiMesh:
    def __init__(self, mesh_path: str=None, index_array_name: str='cell_scalars', roi_ids: list=[1, 8]) -> None:
        self._mesh_pts: pv.UnstructuredGrid = None
        self.__index_array_name: str = index_array_name
        self.__intial_bounds: dict = {}
        self.voxels: pv.UniformGrid = None
        self.affine: np.array = None
        self.intensities: np.array = None
        self._mesh: pv.UnstructuredGrid = self.load_mesh(os.path.realpath(mesh_path), index_array_name, roi_ids) if mesh_path else None

    def load_mesh(self, mesh_path: str, index_array_name: str='cell_scalars', roi_ids: list=[1, 8]) -> None:
        self.__index_array_name = index_array_name
        self._mesh = pv.UnstructuredGrid(os.path.realpath(mesh_path))
        self._mesh_pts = self._mesh.threshold(value=roi_ids, scalars=self.__index_array_name, preference='cell').cell_centers()

        return self._mesh

    def assign_intensities(self, assign_values_per_region: bool, assign_index: bool=False, values_to_assign=None, unwanted_region_ids: list=None):
        roi_pts_loc = np.isin(self._mesh_pts[self.__index_array_name], unwanted_region_ids, invert=True)
        roi_pts_ids = self._mesh_pts[self.__index_array_name][roi_pts_loc]
        roi_intensities = np.zeros(roi_pts_ids.shape)

        self.intensities = np.zeros(self._mesh_pts[self.__index_array_name].shape)
        assert (assign_index == False) and (values_to_assign is not None), 'To assign values a list of those shall be given.'

        if assign_values_per_region == False:
            assert len(roi_intensities) == len(values_to_assign), 'The provided intensity values shall be the same length as the points.'

        for index, id in enumerate(np.unique(roi_pts_ids)):
            pts_loc = np.where(roi_pts_ids == id)[0]
            if assign_index:
                roi_intensities[pts_loc] = 0 if (id in unwanted_region_ids) else id
            else:
                if assign_values_per_region:
                    roi_intensities[pts_loc] = 0 if (id in unwanted_region_ids) else values_to_assign[index]
                else:
                    roi_intensities[pts_loc] = 0 if (id in unwanted_region_ids) else values_to_assign[pts_loc]
        self.intensities[roi_pts_loc] = roi_intensities

        return self.intensities

    def generate_uniform_grid(self, distance_margin: float, voxel_size: float, interp_radius_multiplier: int=4, interp_sharpness: int=8, **kwargs) -> Tuple[pv.UniformGrid, np.array]:
        if self.intensities is None:
            _ = self.assign_intensities(kwargs)
        data = {'x': self._mesh_pts.points[:, 0], 'y': self._mesh_pts.points[:, 1], 'z': self._mesh_pts.points[:, 2], 'intensities': self.intensities}
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'intensities'])
        grid_points = PVGeo.points_to_poly_data(df)

        bounds = self._mesh.bounds
        x_bounds_init = np.ceil(np.sum(np.abs(bounds[0:2]) + distance_margin)/voxel_size).astype(np.int32)
        y_bounds_init = np.ceil(np.sum(np.abs(bounds[2:4]) + distance_margin)/voxel_size).astype(np.int32)
        z_bounds_init = np.ceil(np.sum(np.abs(bounds[4:6]) + distance_margin)/voxel_size).astype(np.int32)
        self.__intial_bounds = {
                'x_bounds': x_bounds_init, 
                'y_bounds': y_bounds_init, 
                'z_bounds': z_bounds_init
            }

        grid = pv.UniformGrid((x_bounds_init, y_bounds_init, z_bounds_init))
        grid.origin = -1*np.array([np.sum(np.abs(bounds[0:2]))/2., np.sum(np.abs(bounds[2:4]))/2., np.sum(np.abs(bounds[4:6]))/2.]) - distance_margin
        grid.spacing = [voxel_size]*3

        self.voxels = grid.interpolate(grid_points, radius=voxel_size*interp_radius_multiplier, sharpness=interp_sharpness)

        ## Affine calculation
        self.affine = self.__calculate_affine(bounds, distance_margin, voxel_size)

        return self.voxels, self.affine

    def __calculate_affine(self, bounds: list, distance_margin: float, voxel_size: float):
        # Get the lower end bounds. The bounds list contains elements as [x-, x+, y-, y+, z-, z+]
        x_minus_bound = bounds[0] - distance_margin
        y_minus_bound = bounds[2] - distance_margin
        z_minus_bound = bounds[4] - distance_margin
        affine = np.array([[voxel_size, 0, 0, x_minus_bound], [0, voxel_size, 0, y_minus_bound], [0, 0, voxel_size, z_minus_bound], [0, 0, 0, 1]])

        return affine

    def generate_nifti(self, image_units: tuple=('mm', 'sec'), save_image: bool=True, image_path: str=None, **knwargs):
        if self.voxels is None:
            _ = self.generate_uniform_grid(knwargs)
        img_header = nib.Nifti1Header()
        img_header.set_xyzt_units(*image_units)
        voxel_data = self.voxels['intensities'].reshape((self.__intial_bounds['z_bounds'], self.__intial_bounds['y_bounds'], self.__intial_bounds['x_bounds'])).transpose()

        img = nib.Nifti1Image(voxel_data, self.affine, img_header)

        if save_image:
            self.save_nifti('converted.nii' if image_path is None else image_path, img)

        return img

    def save_nifti(self, image_path: str, image_object: Union[nib.Nifti1Image, nib.Nifti2Image]):
        nib.save(image_object, os.path.realpath(image_path))
