import os
import gc
import numpy as np
import pandas as pd

def modulation_envelope(e_field_1, e_field_2):
    envelope = np.zeros(e_field_1.shape[0])
    # Calculate the angles between the two fields for each vector
    dot_angle = np.einsum('ij,ij->i', e_field_1, e_field_2)
    cross_angle = np.linalg.norm(np.cross(e_field_1, e_field_2), axis=1)
    angles = np.arctan2(cross_angle, dot_angle)
    # Flip the direction of the electric field if the angle between the two is greater or equal to 90 degrees
    e_field_2 = np.where(np.broadcast_to(angles >= np.pi/2., (3, e_field_2.shape[0])).T, -e_field_2, e_field_2)
    # Recalculate the angles
    dot_angle = np.einsum('ij,ij->i', e_field_1, e_field_2)
    cross_angle = np.linalg.norm(np.cross(e_field_1, e_field_2), axis=1)
    angles = np.arctan2(cross_angle, dot_angle)
    E_minus = np.subtract(e_field_1, e_field_2) # Create the difference of the E fields
    # Condition to have two times the E2 field amplitude
    max_condition_1 = np.linalg.norm(e_field_2, axis=1) < np.linalg.norm(e_field_1, axis=1)*np.cos(angles)
    e1_gr_e2 = np.where(np.linalg.norm(e_field_1, axis=1) > np.linalg.norm(e_field_2, axis=1), max_condition_1, False)
    # Condition to have two times the E1 field amplitude
    max_condition_2 = np.linalg.norm(e_field_1, axis=1) < np.linalg.norm(e_field_2, axis=1)*np.cos(angles)
    e2_gr_e1 = np.where(np.linalg.norm(e_field_2, axis=1) > np.linalg.norm(e_field_1, axis=1), max_condition_2, False)
    # Double magnitudes
    envelope = np.where(e1_gr_e2, 2.0*np.linalg.norm(e_field_2, axis=1), envelope) # 2E2 (First case)
    envelope = np.where(e2_gr_e1, 2.0*np.linalg.norm(e_field_1, axis=1), envelope) # 2E1 (Second case)
    # Calculate the complement area to the previous calculation
    e1_gr_e2 = np.where(np.linalg.norm(e_field_1, axis=1) > np.linalg.norm(e_field_2, axis=1), np.logical_not(max_condition_1), False)
    e2_gr_e1 = np.where(np.linalg.norm(e_field_2, axis=1) > np.linalg.norm(e_field_1, axis=1), np.logical_not(max_condition_2), False)
    # Cross product
    envelope = np.where(e1_gr_e2, 2.0*(np.linalg.norm(np.cross(e_field_2, E_minus), axis=1)/np.linalg.norm(E_minus, axis=1)), envelope) # (First case)
    envelope = np.where(e2_gr_e1, 2.0*(np.linalg.norm(np.cross(e_field_1, -E_minus), axis=1)/np.linalg.norm(-E_minus, axis=1)), envelope) # (Second case)
    return np.nan_to_num(envelope)

UNOPTIMIZED = False
MAX_E_FIELD_CALC = False

if MAX_E_FIELD_CALC:
    def region_calc(modulation_values, aal_regions, region_volumes, regions_of_interest, ommit=None):
        roi = np.isin(aal_regions, regions_of_interest)    
        return np.amax(modulation_values[roi])
else:
    def region_calc(modulation_values, aal_regions, region_volumes, regions_of_interest, ommit=None):
        roi_region_sum = 0
        non_roi_region_sum = 0
        
        for region in np.unique(aal_regions):
            roi = np.where(aal_regions == region)[0]
            #if region in ommit:
            #        continue
            if int(region) in regions_of_interest:
                roi_region_sum += np.sum(modulation_values[roi])/region_volumes[int(region)]
            else:
                non_roi_region_sum += np.sum(modulation_values[roi])/region_volumes[int(region)]
        
        return np.nan_to_num(roi_region_sum/non_roi_region_sum)


if __name__ == "__main__":
    npz_dir = 'directory of the NPZ model files'
    csv_dir = 'directory of the optimized electrode pairs'
    files = next(os.walk(csv_dir))[2]
    
    models = np.sort(next(os.walk(csv_dir))[2])
    sum_values_df = {'TS': []}
    focal_values = {'TS': []}
    focal_values_df = {'TS': []}
    
    if MAX_E_FIELD_CALC:
        mult_factor = 1
    else:
        mult_factor = 10000
    
    for model in models:
        model = model.split('.')[0].split('_')[-1]
        npz_arrays = np.load(os.path.join(npz_dir, model + '_fields_brain_reg.npz'), allow_pickle=True)
        electrode_vals = pd.read_csv(os.path.join(csv_dir, 'optimized_electrodes_' + model + '.csv'))
        
        field_data = npz_arrays['e_field']
        aal_regions = npz_arrays['aal_ids']
        aal_regions[np.isin(aal_regions, np.arange(122, 150, 2))] = 122
        aal_regions[np.isin(aal_regions, np.arange(121, 150, 2))] = 121
        region_volumes = {}
        for region in np.unique(aal_regions):
            region_volumes[region] = np.where(aal_regions == region)[0].size
        
        print(model)
        
        if UNOPTIMIZED:
            electrodes_model = [53, 8, 52, 24]
            currents = [1.25, 1.25, 0.75, 0.75]
        else:
            electrodes_model = electrode_vals['electrodes'].to_numpy().astype(int)
            currents = electrode_vals['currents'].to_numpy()
        
        base_e_field = currents[0]*(field_data[electrodes_model[0]] - field_data[electrodes_model[1]])
        df_e_field = currents[2]*(field_data[electrodes_model[2]] - field_data[electrodes_model[3]])
        
        max_mod = modulation_envelope(base_e_field, df_e_field)
        
        sum_values_df[model] = {}
        for region in np.unique(aal_regions):
            roi = np.where(aal_regions == region)[0]
            sum_values_df[model][str(int(region))] = np.sum(max_mod[roi])/float(roi.size)
        
        ## Right area  
        thal_right = region_calc(max_mod, aal_regions, region_volumes, np.arange(122, 150, 2), [42])
        hip_right = region_calc(max_mod, aal_regions, region_volumes, [42], np.arange(122, 150, 2))
        focal_values[model] = {'Thal_Right': thal_right*mult_factor}
        focal_values[model]['Hipp_Right'] = hip_right*mult_factor
        ## Right area
        
        ## Left area      
        thal_left = region_calc(max_mod, aal_regions, region_volumes, np.arange(121, 150, 2), [41])
        hip_left = region_calc(max_mod, aal_regions, region_volumes, [41], np.arange(121, 150, 2))
        focal_values[model]['Thal_Left'] = thal_left*mult_factor
        focal_values[model]['Hipp_Left'] = hip_left*mult_factor
        ## Left area
        
        ## Both areas
        thal_both = region_calc(max_mod, aal_regions, region_volumes, np.arange(121, 151, 1), [41, 42])
        hip_both = region_calc(max_mod, aal_regions, region_volumes, [41, 42], np.arange(121, 151, 1))
        focal_values[model]['Thal_Both'] = thal_both*mult_factor
        focal_values[model]['Hipp_Both'] = hip_both*mult_factor
        ## Both areas
        
        ## Cortex
        cortex = region_calc(max_mod, aal_regions, region_volumes, [-4], [-5])
        wm = region_calc(max_mod, aal_regions, region_volumes, [-5], [-4])
        focal_values[model]['Cortex'] = cortex*mult_factor
        focal_values[model]['White_Matter'] = wm*mult_factor
        ## Cortex
        
        del npz_arrays
        del field_data
        del base_e_field
        del df_e_field
        del max_mod
        gc.collect()

sum_values_df.pop('TS')
focal_values.pop('TS')

np.savez('input the save path', sum_values_df=sum_values_df, focal_values=focal_values)

