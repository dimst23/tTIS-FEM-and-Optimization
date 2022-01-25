import os
import gc
import sys
import yaml
import time
import numpy as np
import pandas as pd
import cupy as cp
from geneticalgorithm import geneticalgorithm as ga

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
parser.add_argument('--npz-file', metavar='str', type=str,
                    action='store', dest='npz_file',
                    default='real_brain', help=helps['model'], required=True)
parser.add_argument('--save-dir', metavar='str', type=str,
                    action='store', dest='save_dir',
                    default=None, required=True)
options = parser.parse_args()
#### Argument parsing

with open(os.path.realpath(options.settings_file)) as stream:
    settings = yaml.safe_load(stream)

if os.name == 'nt':
    extra_path = '_windows'
else:
    extra_path = ''

sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))

from modulation_envelope import modulation_envelope_gpu

def objective_df(x, field_data, regions_of_interest, aal_regions, region_volumes, currents, threshold):  
    if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
        return 100*(np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000
    
    penalty = 0
    electrodes = np.round(x[:4]).astype(np.int32) # The first 4 indices are the electrode IDs
    roi = cp.isin(aal_regions, cp.array(regions_of_interest))
    max_vals = []
    fitness_vals = []
    
    for current in currents:
        e_field_base = current[0]*field_data[electrodes[0]] - current[0]*field_data[electrodes[1]]
        e_field_df = current[1]*field_data[electrodes[2]] - current[1]*field_data[electrodes[3]]

        e_field_base_gpu = cp.array(e_field_base)
        e_field_df_gpu = cp.array(e_field_df)
        modulation_values = modulation_envelope_gpu(e_field_base_gpu, e_field_df_gpu)
        max_vals.append(float(cp.amax(modulation_values[roi])))
    
        roi_region_sum = 0
        non_roi_region_sum = 0

        for region in cp.unique(aal_regions):
            roi = cp.where(aal_regions == region)[0]
        
            if int(region) in regions_of_interest:
                roi_region_sum += cp.sum(modulation_values[roi])/region_volumes[int(region)]
            else:
                non_roi_region_sum += cp.sum(modulation_values[roi])/region_volumes[int(region)]
        
        region_ratio = cp.nan_to_num(roi_region_sum/non_roi_region_sum)
        fitness_measure = region_ratio*10000
        fitness_vals.append(float(fitness_measure))
    
    max_vals = np.array(max_vals)
    max_val_curr = np.where(max_vals >= threshold)[0]
    fitness_vals = np.array(fitness_vals)

    return_fitness = 0
    if max_val_curr.size == 0:
        penalty += 100*((threshold - np.mean(max_vals))**2) + 1000
        return_fitness = np.amin(fitness_vals)
    else:
        fitness_candidate = np.amax(fitness_vals[max_val_curr])
        return_fitness = fitness_candidate
    
    return -float(np.round(return_fitness - penalty, 2))

if __name__ == "__main__":
    OPTIMIZATION_THRESHOLD = 0.2
    
    npz_arrays = np.load(options.npz_file, allow_pickle=True)
    
    field_data = npz_arrays['e_field']
    aal_regions = npz_arrays['aal_ids']
    region_volumes = {}
    for region in np.unique(aal_regions):
        region_volumes[region] = np.where(aal_regions == region)[0].size
    roi_ids = np.array([42])
    
    algorithm_param = {'max_num_iteration': 25,
                       'population_size': 100,
                       'mutation_probability': 0.4,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.1,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None
                    }
    
    cur_potential_values = np.arange(0.5, 1.55, 0.05)
    cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
    cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))
    usable_currents = cur_all_combinations[np.where(np.sum(np.round(cur_all_combinations, 2), axis=1) == 2)[0]]
    
    aal_regions_gpu = cp.array(aal_regions)
    ga_objective_df = lambda x, **kwargs: objective_df(x, field_data, roi_ids, aal_regions=aal_regions_gpu, region_volumes=region_volumes, currents=usable_currents, threshold=OPTIMIZATION_THRESHOLD)
    
    var_type = np.array([['int']]*4)
    bounds = np.array([[0, 60]]*4)
    result = ga(function=ga_objective_df, dimension=bounds.shape[0], variable_type_mixed=var_type, variable_boundaries=bounds, algorithm_parameters=algorithm_param, function_timeout=120., convergence_curve=False)
    result.run()

    convergence = result.report
    solution = result.output_dict

    model_id = os.path.basename(options.npz_file).split('.')[0].split('_')[0]
    df_dict = {'electrodes': solution['variable'], 'value': solution['function']}
    df = pd.DataFrame(df_dict)

    df.to_csv(os.path.join(options.save_dir, 'optimized_electrodes_' + model_id + '.csv'))
