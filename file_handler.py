import constants as const
import pandas as pd
import os

def save_results_to_file(data: pd.DataFrame):
    try:
        os.makedirs(const.RESULTS_DIR)
    except:
        pass
    if os.path.isfile(const.RESULTS_FILE_LOC):
        __move_results_to_new_subdirectory()
    
    print(f'Writing results to {const.RESULTS_FILE_LOC}')
    data.to_csv(const.RESULTS_FILE_LOC, index=False)
    
def read_from_subdirectory(subdirectory: str = '') -> pd.DataFrame:
    path: str = os.path.join(const.RESULTS_DIR, subdirectory, const.RESULTS_FILE_NAME) if subdirectory else const.RESULTS_FILE_LOC
    try:
        return pd.read_csv(path)
    except:
        raise IOError(f'{path} cannot be read.')
    
def full_path_from_subdirectory(subdirectory: str, file_name: str) -> str:
    path: str = os.path.join(const.RESULTS_DIR, subdirectory, file_name) if subdirectory else os.path.join(const.RESULTS_DIR, file_name)
    return path

def __move_results_to_new_subdirectory():
    min_unused_folder = 1
    while True:
        new_path = os.path.join(const.RESULTS_DIR, f'Results_{min_unused_folder}')
        try:
            os.makedirs(new_path)
            break
        except:
            min_unused_folder += 1
    print(f'Moving existing results to {new_path}')
    for f in os.listdir(const.RESULTS_DIR):
        if os.path.isfile(os.path.join(const.RESULTS_DIR, f)):
            os.rename(os.path.join(const.RESULTS_DIR, f), os.path.join(new_path, f))