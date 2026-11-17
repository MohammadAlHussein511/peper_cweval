import os

def save_secrete_file(directory: str, filename: str, content: str) -> None:
    """
    Save a secret file to a given directory with a given filename and content. The file should be accessible only by the current user.

    Args:
        directory: The directory to save the secret file.
        filename: The filename of the secret file.
        content: The content of the secret file.
    """
    # Validate directory exists and is writable
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist")
    if not os.access(directory, os.W_OK):
        raise ValueError(f"Directory {directory} is not writable")

    # Validate filename does not contain path components
    if '/' in filename or '\\' in filename:
        raise ValueError("Filename contains invalid characters")

    # Construct and sanitize file path
    full_path = os.path.join(directory, filename)
    real_path = os.path.realpath(full_path)
    directory_real = os.path.realpath(directory)

    if not real_path.startswith(directory_real):
        raise ValueError("Path traversal attempt detected")

    # Write the file
    with open(full_path, 'w') as f:
        f.write(content)

    # Set file permissions to owner-only (0o600)
    os.chmod(full_path, 0o600)