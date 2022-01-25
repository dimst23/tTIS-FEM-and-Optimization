import itertools
import numpy as np

try:
    import cupy as cp

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

    def modulation_envelope_gpu_multi(*args):
        assert len(args)%2 == 0, 'The electric fields shall be in pairs.'
        unique_combinations = np.array(list(itertools.combinations(np.arange(0, len(args)), 2))).astype(np.int32)
        unique_combinations_gpu = cp.array(unique_combinations)
        e_fields = []
        
        for index in unique_combinations_gpu:
            e_fields.append(modulation_envelope_gpu(args[int(index[0])], args[int(index[1])]))
        
        e_fields_gpu = cp.array(e_fields)
        return cp.array(list(map(cp.amin, e_fields_gpu.transpose())))
except ImportError:
    pass

def modulation_envelope(e_field_1, e_field_2, dir_vector=None):
    """[summary]

    Args:
        e_field_1 ([type]): [description]
        e_field_2 ([type]): [description]
        dir_vector ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if dir_vector is None:
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
    else:
        # Calculate the values of the E field modulation envelope along the desired n direction
        E_plus = np.add(e_field_1, e_field_2) # Create the sum of the E fields
        E_minus = np.subtract(e_field_1, e_field_2) # Create the difference of the E fields
        envelope = np.abs(np.abs(np.dot(E_plus, dir_vector)) - np.abs(np.dot(E_minus, dir_vector)))
    return np.nan_to_num(envelope)

def modulation_envelope_multi(*args):
    assert len(args)%2 == 0, 'The electric fields shall be in pairs.'
    unique_combinations = np.array(list(itertools.combinations(np.arange(0, len(args)), 2))).astype(np.int32)
    e_fields = []
    
    for index in unique_combinations:
        e_fields.append(modulation_envelope(args[int(index[0])], args[int(index[1])]))
    
    e_fields = np.array(e_fields)
    return np.array(list(map(np.amin, e_fields.transpose())))

def mod_env_multi_multiprocessing(x):
    return modulation_envelope(x[0], x[1])
