import os

def read_file_from_dir(filename: str, dir_path: str) -> str:
    '''
    Read the content of the file with a user-provided filename from the given directory path. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

    Args:
        filename (str): The name of the file to read, requested by a user.
        dir_path (str): The directory path where the file is located.

    Returns:
        str: The content of the file, or an empty string if any error occurs.
    '''
    # Validate filename to prevent path traversal
    if '/' in filename or '\\' in filename:
        return ""
    
    # Construct the full path
    full_path = os.path.join(dir_path, filename)
    
    # Validate that the full path is within the dir_path
    try:
        abs_dir = os.path.abspath(dir_path)
        abs_full = os.path.abspath(full_path)
        if not abs_full.startswith(abs_dir):
            return ""
    except:
        return ""
    
    # Attempt to read the file
    try:
        with open(full_path, 'r') as f:
            return f.read()
    except:
        return ""
