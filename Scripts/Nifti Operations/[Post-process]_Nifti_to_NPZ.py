import os
import gc
import yaml
import numpy as np
import nibabel as nib

from argparse import ArgumentParser, RawDescriptionHelpFormatter

#### Argument parsing
helps = {
    'settings-file' : "File having the settings to be loaded",
    'aal-atlas' : "Full path of the AAL atlas Nifti",
    'model-path' : "Full path of the dirctory which includes the models",
}

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('--settings-file', metavar='str', type=str,
                    action='store', dest='settings_file',
                    default=None, help=helps['settings-file'], required=True)
parser.add_argument('--aal-atlas', metavar='str', type=str,
                    action='store', dest='atlas_path',
                    default=None, help=helps['aal-atlas'], required=True)
parser.add_argument('--model-path', metavar='str', type=str,
                    action='store', dest='model_path',
                    default='real_brain', help=helps['model-path'], required=True)
parser.add_argument('--save-dir', metavar='str', type=str,
                    action='store', dest='save_dir',
                    default=None, required=True)
options = parser.parse_args()
#### Argument parsing

folders = np.sort(next(os.walk(options.model_path))[1])
e_field_size = []

with open(os.path.realpath(options.settings_file)) as stream:
    settings = yaml.safe_load(stream)

atlas_nifti = nib.load(options.atlas_path)
aal_regions_loc = np.where(atlas_nifti.get_fdata().flatten() != 0)[0].astype(int)
aal_regions_ids = atlas_nifti.get_fdata().flatten()[aal_regions_loc]

settings_electrodes = settings['SfePy']['electrodes']['10-10-mod'].keys()
electrode_constant = 10

for fld in folders:
    e_field_values = []
    model_id = fld
    print(model_id)
    
    for electrode in settings_electrodes:
        electrode_name = electrode
        if electrode_name == 'P9':
            continue
        print(electrode_name)
        files = np.sort(next(os.walk(os.path.join(options.model_path, model_id, electrode_name)))[2])
        
        nifti_images_dict = {}
        for fl in files:
            if not fl.startswith('w'):
                continue
            field_montage = fl.split('.')[0].split('_')[2]
            field_direction = fl.split('.')[0].split('_')[1]
            dict_key = '{}_{}'.format(field_montage, field_direction)
            
            nifti_images_dict[dict_key] = nib.load(os.path.join(options.model_path, model_id, electrode_name, fl))
        
        e_field = np.nan_to_num(np.vstack((nifti_images_dict[electrode_name + '_x'].get_fdata().flatten(), nifti_images_dict[electrode_name + '_y'].get_fdata().flatten(), nifti_images_dict[electrode_name + '_z'].get_fdata().flatten())).transpose())
        e_field = e_field[aal_regions_loc].reshape((1, -1, 3))
        
        if isinstance(e_field_values, list):
            e_field_values = e_field
        else:
            e_field_values = np.append(e_field_values, e_field, axis=0)
        
        del nifti_images_dict
        del e_field
        gc.collect()

    np.savez_compressed(os.path.join(options.save_dir, model_id + '_fields_brain_reg'), e_field=e_field_values.reshape((4, -1, 3)), aal_ids=aal_regions_ids, aal_loc=aal_regions_loc)

    del e_field_values
    gc.collect()
