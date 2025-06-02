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


def load_glmsingle_outputs_h5(filename):
    """
    Robust HDF5 loading function for GLMsingle outputs with support for ragged nested lists.
    
    This function reconstructs the original data structure by handling:
    - Regular numpy arrays (loaded from datasets)
    - Lists of uniform-shaped arrays (unstacked from datasets)  
    - Lists of ragged arrays (loaded from groups with individual datasets)
    - None values (loaded from empty datasets)
    - Scalars (loaded from attributes)
    
    Args:
        filename (str): Path to the HDF5 file to load
        
    Returns:
        dict: Dictionary containing the reconstructed GLMsingle output data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        OSError: If the file is not a valid HDF5 file
    """
    outdict = {}

    try:
        with h5py.File(filename, 'r') as f:
            # Load attributes (scalars)
            for key, value in f.attrs.items():
                outdict[key] = value
            
            # Load datasets and groups
            for key in f.keys():
                item = f[key]
                
                if isinstance(item, h5py.Dataset):
                    # Handle empty datasets (None values) - check shape/size, not dtype
                    if item.shape is None or item.size is None:
                        outdict[key] = None
                    else:
                        # Regular numpy array
                        outdict[key] = np.array(item)
                        
                elif isinstance(item, h5py.Group):
                    # Check if this is a ragged list group
                    if '__glmsingle_ragged_list__' in item.attrs:
                        print('hi')
                        # Reconstruct ragged list of arrays
                        ragged_list = []
                        # Sort by numeric key to maintain order
                        sorted_keys = sorted(item.keys(), key=int)
                        for sub_key in sorted_keys:
                            ragged_list.append(np.array(item[sub_key]))
                        outdict[key] = ragged_list
                    else:
                        # Handle other group types if needed
                        print(f"Warning: Unrecognized group structure for key '{key}'")
                        outdict[key] = None
                        
    except FileNotFoundError:
        raise FileNotFoundError(f"HDF5 file not found: {filename}")
    except OSError as e:
        raise OSError(f"Error reading HDF5 file '{filename}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading GLMsingle outputs from '{filename}': {e}")
    
    return outdict
                