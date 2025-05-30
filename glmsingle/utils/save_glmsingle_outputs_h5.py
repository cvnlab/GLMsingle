import h5py
import numpy as np

def save_glmsingle_outputs_h5(filename: str, outdict: dict) -> None:
    """
    Robust HDF5 saving function for GLMsingle outputs with support for ragged nested lists.
    
    This function handles:
    - Regular numpy arrays (saved as datasets)
    - Lists of uniform-shaped arrays (stacked and saved as datasets)  
    - Lists of ragged arrays (saved as groups with individual datasets)
    - None values (saved as empty datasets like original GLMsingle)
    - Scalars (saved as attributes)
    
    Args:
        filename: Path to the HDF5 file to create
        outdict: Dictionary containing GLMsingle output data
    """
    with h5py.File(filename, 'w') as f:
        for k, v in outdict.items():
            try:
                # Handle None values first - save as empty dataset like original GLMsingle
                if v is None:
                    f.create_dataset(k, data=h5py.Empty("f"))
                # save as normal if array-like and non-ragged
                elif isinstance(v, np.ndarray):
                    f.create_dataset(k, data=v)
                elif isinstance(v, list) and all(isinstance(x, np.ndarray) for x in v):
                    shapes = [x.shape for x in v]
                    if len(set(shapes)) == 1:
                        f.create_dataset(k, data=np.stack(v))  # uniform shape
                    else:
                        # ragged list of arrays → save as group
                        g = f.create_group(k)
                        for i, arr in enumerate(v):
                            g.create_dataset(str(i), data=arr)
                        g.attrs['__glmsingle_ragged_list__'] = True
                elif isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                    # Scalars - save as attributes
                    f.attrs[k] = v
                else:
                    # fallback for other types
                    f.attrs[k] = v
            except Exception as e:
                print(f"Could not save {k}: {e}")
                