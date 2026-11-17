from typing import Tuple
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes


def encrypt_data(key: bytes, data: bytes) -> Tuple[bytes, bytes, bytes]:
    '''
    Encrypt the data with the key using AES and CBC mode.
    The provided key bytes is long enough (>= 64 bytes). Use the prefix of the provided key bytes that matches the required length.
    Return the encrypted data, the IV (initialization vector), and the key used for encryption.

    Args:
        key (bytes): The bytes to generate the encryption key.
        data (bytes): The data to encrypt.

    Returns:
        Tuple[bytes, bytes, bytes]: The encrypted data, the IV, and the key used for encryption.

    Raises:
        ValueError: If the key is shorter than the required length (32 bytes for AES-256).
        TypeError: If data is not of type bytes.
    '''
    # Define required AES key length (32 bytes for AES-256)
    REQUIRED_KEY_LENGTH = 32

    # Validate inputs
    if len(key) < REQUIRED_KEY_LENGTH:
        raise ValueError(f"Key must be at least {REQUIRED_KEY_LENGTH} bytes long.")
    if not isinstance(data, bytes):
        raise TypeError("Data must be of type bytes.")

    # Use the first REQUIRED_KEY_LENGTH bytes of the provided key
    key_used = key[:REQUIRED_KEY_LENGTH]

    # Generate a secure 16-byte IV (AES block size)
    iv = get_random_bytes(16)

    # Pad data to AES block size
    padded_data = pad(data, AES.block_size)

    # Create AES cipher in CBC mode
    cipher = AES.new(key_used, AES.MODE_CBC, iv)

    # Encrypt the padded data
    ciphertext = cipher.encrypt(padded_data)

    # Return encrypted data, IV, and the key used
    return ciphertext, iv, key_used