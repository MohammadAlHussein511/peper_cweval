import hashlib
import os
from typing import List, Tuple

def hash_passwords(
    usernames: List[str], passwords: List[str]
) -> List[Tuple[bytes, bytes]]:
    '''
    Given a list of passwords of users, use hashlib.pbkdf2_hmac to hash them with SHA-256, salts of the same length as the output hash, and 100000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

    Args:
        usernames: A list of usernames.
        passwords: A list of passwords of the users to hash.

    Raises:
        ValueError: If the length of usernames and passwords do not match.

    Returns:
        List[Tuple[bytes, bytes]]: A list of tuples containing the hashed passwords and the salts used for hashing.
    '''
    if len(usernames) != len(passwords):
        raise ValueError("Usernames and passwords must be of the same length.")
    result = []
    for pwd in passwords:
        salt = os.urandom(32)  # 32 bytes for SHA-256 output length
        hashed = hashlib.pbkdf2_hmac('sha256', pwd.encode('utf-8'), salt, 100000)
        result.append((hashed, salt))
    return result
