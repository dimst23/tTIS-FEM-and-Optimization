from __future__ import absolute_import
import os
import gc
import sys
import yaml
import numpy as np
import pyvista as pv

from argparse import ArgumentParser, RawDescriptionHelpFormatter

#### Argument parsing
helps = {
    'settings-file' : "File having the settings to be loaded",
    'model' : "Name of the model. Selection from the settings file",
}

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('--settings-file', metavar='str', type=str,
                    action='store', dest='settings_file',
                    default=None, help=helps['settings-file'], required=True)
parser.add_argument('--meshf', metavar='str', type=str,
                    action='store', dest='meshf',
                    default=None, required=True)
parser.add_argument('--model', metavar='str', type=str,
                    action='store', dest='model',
                    default='real_brain', help=helps['model'], required=True)
parser.add_argument('--csv_save_dir', metavar='str', type=str,
                    action='store', dest='csv_save_dir',
                    default=None, required=True)
parser.add_argument('--job_id', metavar='str', type=str,
                    action='store', dest='job_id',
                    default='', required=False)
options = parser.parse_args()
#### Argument parsing

with open(os.path.realpath(options.settings_file)) as stream:
    settings = yaml.safe_load(stream)

if os.name == 'nt':
    extra_path = '_windows'
else:
    extra_path = ''

sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))

import FEM.Solver as slv

settings['SfePy'][options.model]['mesh_file' + extra_path] = options.meshf
model_id = os.path.basename(options.meshf).split('.')[0].split('_')[-1]

## Read the mesh with pyvista to get the area ids and AAL regions
msh = pv.UnstructuredGrid(options.meshf)
brain_regions_mask = np.isin(msh['cell_scalars'], [4, 5, 6])

cell_ids_brain = msh['cell_scalars'][brain_regions_mask]
aal_regions = msh['AAL_regions'][brain_regions_mask]
cell_volumes_brain = msh.compute_cell_sizes().cell_arrays['Volume'][brain_regions_mask]

region_volumes_brain = {}
for region in np.unique(aal_regions):
    roi = np.where(aal_regions == region)[0]
    region_volumes_brain[int(region)] = np.sum(cell_volumes_brain[roi])
del msh
region_volumes_brain = np.array(region_volumes_brain)
## Read the mesh with pyvista to get the area ids and AAL regions

electrodes = settings['SfePy']['electrodes']['10-10-mod']
e_field_values_brain = []

solve = slv.Solver(settings, 'SfePy', '10-10-mod')
solve.load_mesh(options.model)

for electrode in electrodes.items():
    if electrode[0] == 'P9':
        continue
    solve.essential_boundaries.clear()
    solve.fields.clear()
    solve.field_variables.clear()

    solve.define_field_variable('potential', 'voltage', out_of_range_assign_region='Skin', out_of_range_group_threshold=71)

    solve.define_essential_boundary(electrode[0], electrode[1]['id'], 'potential', current=1)
    solve.define_essential_boundary('P9', 71, 'potential', current=-1)

    solve.solver_setup(600, 1e-12, 5e-12, verbose=True)
    solution = solve.run_solver(save_results=False, post_process_calculation=True)

    e_field_base = solution['e_field_(potential)'].data[:, 0, :, 0]
    if isinstance(e_field_values_brain, list):
        e_field_values_brain = e_field_base[brain_regions_mask]
    else:
        e_field_values_brain = np.append(e_field_values_brain, e_field_base[brain_regions_mask], axis=0)

    del solution
    gc.collect()

del solve
gc.collect

np.savez_compressed(os.path.join(options.csv_save_dir, model_id + '_fields_brain'), e_field=e_field_values_brain.reshape((61, -1, 3)), cell_ids=cell_ids_brain, aal_regions=aal_regions, volumes=region_volumes_brain)
