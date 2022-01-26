import os
import gc
import numpy as np
import cupy as cp
import pandas as pd

def modulation_envelope_gpu(e_field_1, e_field_2, dir_vector=None):
    if dir_vector is None:
        envelope = cp.zeros(e_field_1.shape[0])
        
        # Calculate the angles between the two fields for each vector
        dot_angle = cp.einsum('ij,ij->i', e_field_1, e_field_2)
        cross_angle = cp.linalg.norm(cp.cross(e_field_1, e_field_2), axis=1)
        angles = cp.arctan2(cross_angle, dot_angle)
        
        # Flip the direction of the electric field if the angle between the two is greater or equal to 90 degrees
        e_field_2 = cp.where(cp.broadcast_to(angles >= cp.pi/2., (3, e_field_2.shape[0])).T, -e_field_2, e_field_2)
        
        # Recalculate the angles
        dot_angle = cp.einsum('ij,ij->i', e_field_1, e_field_2)
        cross_angle = cp.linalg.norm(cp.cross(e_field_1, e_field_2), axis=1)
        angles = cp.arctan2(cross_angle, dot_angle)
        E_minus = cp.subtract(e_field_1, e_field_2) # Create the difference of the E fields
        
        # Condition to have two times the E2 field amplitude
        max_condition_1 = cp.linalg.norm(e_field_2, axis=1) < cp.linalg.norm(e_field_1, axis=1)*cp.cos(angles)
        e1_gr_e2 = cp.where(cp.linalg.norm(e_field_1, axis=1) > cp.linalg.norm(e_field_2, axis=1), max_condition_1, False)
        
        # Condition to have two times the E1 field amplitude
        max_condition_2 = cp.linalg.norm(e_field_1, axis=1) < cp.linalg.norm(e_field_2, axis=1)*cp.cos(angles)
        e2_gr_e1 = cp.where(cp.linalg.norm(e_field_2, axis=1) > cp.linalg.norm(e_field_1, axis=1), max_condition_2, False)
        
        # Double magnitudes
        envelope = cp.where(e1_gr_e2, 2.0*cp.linalg.norm(e_field_2, axis=1), envelope) # 2E2 (First case)
        envelope = cp.where(e2_gr_e1, 2.0*cp.linalg.norm(e_field_1, axis=1), envelope) # 2E1 (Second case)
        
        # Calculate the complement area to the previous calculation
        e1_gr_e2 = cp.where(cp.linalg.norm(e_field_1, axis=1) > cp.linalg.norm(e_field_2, axis=1), cp.logical_not(max_condition_1), False)
        e2_gr_e1 = cp.where(cp.linalg.norm(e_field_2, axis=1) > cp.linalg.norm(e_field_1, axis=1), cp.logical_not(max_condition_2), False)
        
        # Cross product
        envelope = cp.where(e1_gr_e2, 2.0*(cp.linalg.norm(cp.cross(e_field_2, E_minus), axis=1)/cp.linalg.norm(E_minus, axis=1)), envelope) # (First case)
        envelope = cp.where(e2_gr_e1, 2.0*(cp.linalg.norm(cp.cross(e_field_1, -E_minus), axis=1)/cp.linalg.norm(-E_minus, axis=1)), envelope) # (Second case)
    else:
        # Calculate the values of the E field modulation envelope along the desired n direction
        E_plus = cp.add(e_field_1, e_field_2) # Create the sum of the E fields
        E_minus = cp.subtract(e_field_1, e_field_2) # Create the difference of the E fields
        envelope = cp.abs(cp.abs(cp.dot(E_plus, dir_vector)) - cp.abs(cp.dot(E_minus, dir_vector)))
    
    return cp.nan_to_num(envelope)


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
    npz_dir = 'directory of the NPZ model files'
    csv_dir = 'directory of the optimized electrode pairs'
    files = next(os.walk(csv_dir))[2]
    
    for model in files:
        model = model.split('.')[0].split('_')[-1]
        
        npz_arrays = np.load(os.path.join(npz_dir, model + '_fields_brain_reg.npz'), allow_pickle=True)
        electrode_vals = pd.read_csv(os.path.join(csv_dir, 'optimized_electrodes_' + model + '.csv'))
        
        field_data = npz_arrays['e_field']
        aal_regions = npz_arrays['aal_ids']
        region_volumes = {}
        for region in np.unique(aal_regions):
            region_volumes[region] = np.where(aal_regions == region)[0].size
        roi_ids = np.array([42])
        
        ideal_case = None
        
        cur_potential_values = np.arange(0.5, 1.55, 0.05)
        cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
        cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))
        usable_currents = cur_all_combinations[np.where(np.sum(np.round(cur_all_combinations, 2), axis=1) == 2)[0]]
        
        print(model)
        electrodes_model = electrode_vals['electrodes'].to_numpy().astype(int)
        aal_regions_gpu = cp.array(aal_regions)
        value, cur_id = objective_df(electrodes_model, field_data, roi_ids, aal_regions=aal_regions_gpu, region_volumes=region_volumes, currents=usable_currents, ideal_case=ideal_case)
        
        electrode_vals['currents'] = np.round([usable_currents[cur_id][0], usable_currents[cur_id][0], usable_currents[cur_id][1], usable_currents[cur_id][1]], 2)
        electrode_vals.to_csv(os.path.join(csv_dir, 'optimized_electrodes_' + model + '.csv'))
        print('Value: {}, Current: {}, {}'.format(value, usable_currents[cur_id][0], usable_currents[cur_id][1]))
        
        del field_data
        del npz_arrays
        gc.collect()
