import pandas as pd
import h5py
import numpy as np

def add_normals_to_traces():
    # Load traces
    traces_path = "storage/output/ground_truth_traces.csv"
    traces = pd.read_csv(traces_path)
    
    # Load DFN HDF5
    hdf5_path = "storage/data/dfn_export_for_python.h5"
    with h5py.File(hdf5_path, 'r') as f:
        normals = f['/fractures/normals'][:]
        
    # Map parent_fracture_id to normals
    # parent_fracture_id in traces is 1-based or 0-based? Let's check:
    # We can check max parent_fracture_id in traces and shape of normals
    max_id = traces['parent_fracture_id'].max()
    print(f"Max parent_fracture_id in traces: {max_id}")
    print(f"Shape of normals in H5: {normals.shape}")
    
    # In generate_dfn.py, parent_fracture_id is set to the 1-based index or 0-based?
    # Let's write safety code: if max_id is larger or equal to normals.shape[0], parent_fracture_id might be 1-based or different.
    # Actually, in generate_dfn.py:
    # ss = all_set_ids[mask]
    # And parent_fracture_id is not explicitly exported in generate_dfn.py's H5 but it is in trace pipeline.
    # Let's map it. Since the indices are from the generated DFN, they are likely 0-based indices or 1-based.
    # Let's check if max_id is < normals.shape[0] (0-based) or not.
    if max_id >= normals.shape[0]:
        print("Warning: parent_fracture_id seems to exceed normals length. We will check if it is 1-based.")
        # If it's 1-based, we subtract 1.
        parent_indices = (traces['parent_fracture_id'] - 1).astype(int)
    else:
        parent_indices = traces['parent_fracture_id'].astype(int)
        
    trace_normals = normals[parent_indices]
    
    traces['normal_x'] = trace_normals[:, 0]
    traces['normal_y'] = trace_normals[:, 1]
    traces['normal_z'] = trace_normals[:, 2]
    
    output_path = "storage/output/ground_truth_traces_with_normals.csv"
    traces.to_csv(output_path, index=False)
    print(f"Successfully wrote traces with normals to: {output_path}")

if __name__ == '__main__':
    add_normals_to_traces()
