import os
import yaml
import pymesh
import Meshing.MeshOperations as MeshOps

import scipy.io as sio
import numpy as np

from argparse import ArgumentParser, RawDescriptionHelpFormatter

#### Argument parsing
helps = {
    'settings-file' : "File having the settings to be loaded",
    'models-dir' : "Full path of the model directory",
    'electrode-attributes-path' : "Full path of the Mesh2EEG generated electrodes file",
}

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('--settings-file', metavar='str', type=str,
                    action='store', dest='settings_file',
                    default=None, help=helps['settings-file'], required=True)
parser.add_argument('--models-dir', metavar='str', type=str,
                    action='store', dest='stl_models_path',
                    default=None, help=helps['models-dir'], required=True)
parser.add_argument('--electrode-attributes-path', metavar='str', type=str,
                    action='store', dest='electrode_attributes_path',
                    default='real_brain', help=helps['electrode-attributes-path'], required=True)
parser.add_argument('--save-dir', metavar='str', type=str,
                    action='store', dest='save_dir',
                    default=None, required=True)
options = parser.parse_args()
#### Argument parsing

folders = np.sort(next(os.walk(options.stl_models_path))[1])

# Electrodes to ommit based on the imported settings file
electrodes_to_omit=['Nz', 'N2', 'AF10', 'F10', 'FT10', 'T10(M2)', 'TP10', 'PO10', 'I2', 'Iz', 'I1', 'PO9', 'TP9', 'T9(M1)', 'FT9', 'F9', 'AF9', 'N1', 'P10']

with open(os.path.realpath(options.settings_file)) as stream:
    settings = yaml.safe_load(stream)

if __name__ == '__main__':
    for folder in folders:
        if folder == 'meshed':
            continue
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(options.electrode_attributes_path, '10-10_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
            'coordinates': standard_electrodes['ElectrodePts'],
            'ids': settings['SfePy']['electrodes']['10-10-mod'],
            'width': 4,
            'radius': 4,
            'elements': 200,
        }

        skin_stl = pymesh.load_mesh(os.path.join(options.stl_models_path, folder, 'skin_fixed.stl'))

        meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
        meshing.load_surface_meshes(os.path.join(options.stl_models_path, folder), ['skin_fixed.stl', 'skull_fixed.stl', 'csf_fixed.stl', 'gm_fixed.stl', 'wm_fixed.stl', 'cerebellum_fixed.stl', 'ventricles_fixed.stl'])
        meshing.phm_model_meshing(os.path.join(options.save_dir, 'meshed_model_10-10_' + folder + '.poly'), electrodes_to_omit=electrodes_to_omit)
