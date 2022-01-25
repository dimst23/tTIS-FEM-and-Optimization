import os
import yaml
import multiprocessing

import NiiPHM.NiiMesh as NiiMesh


with open(os.path.realpath('settings file path')) as stream:
    settings = yaml.safe_load(stream)

index_array_name = 'cell_scalars'

model_path_vtk = 'full path of the directory containing the VTK solved models'
files = next(os.walk(model_path_vtk))[2]

conductivities = [1 for a in settings['SfePy']['real_brain']['regions'].values()]

def nfts(fl):
    global conductivities
    model_path_vtk = 'full path of the directory containing the VTK solved models'
    index_array_name = 'cell_scalars'
    structures_to_remove = [1, 2, 3, 5, 6, 7]
    
    output_dir = 'output directory to save the Nifti files'
    model_id = os.path.splitext(fl)[0].split('_')[-1].split('-')[0].split('.')[0]
    print("Converting model {}".format(model_id))
    
    nifti = NiiMesh.NiiMesh(os.path.join(model_path_vtk, fl), index_array_name, [1, 8])
    _ = nifti.assign_intensities(assign_index=False, unwanted_region_ids=structures_to_remove, assign_values_per_region=True, values_to_assign=conductivities)
    _ = nifti.generate_uniform_grid(15, 0.75, )
    nifti.generate_nifti(image_path=os.path.join(output_dir, 'PHM_' + model_id + '.nii'))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(nfts, files)
