import hashlib
import secrets


def hash_certificate(cert: str) -> str:
    """
    Hash the given certificate securely and return the hash as a hexadecimal string.

    Args:
        cert (str): The certificate to be hashed.

    Returns:
        str: The securely hashed certificate as a hexadecimal string.
    """
    # Generate a cryptographically secure random salt
    salt = secrets.token_bytes(16)
    # Combine the salt with the certificate data
    combined = salt + cert.encode('utf-8')
    # Hash the combined data using SHA-256
    return hashlib.sha256(combined).hexdigest()