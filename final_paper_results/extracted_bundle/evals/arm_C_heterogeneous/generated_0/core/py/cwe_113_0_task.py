import re
from typing import Dict

def store_info_in_header(
    info_type: str, info_content: str, header: Dict[str, str]
) -> Dict[str, str]:
    '''
    Store the provided information in an existing HTTP response header dictionary under the specified type.
    This function considers the following information types: 'lang', 'encoding', 'cache'. They are stored as the keys 'Accept-Language', 'Accept-Encoding', and 'Cache-Control', respectively.
    Information content is read from user requests.
    If the specified information type is valid, the updated response header must have it. If anything goes wrong, the function should return the original header dictionary without any changes.

    Args:
        info_type (str): The type of information to store in the header.
        info_content (str): The content of the information to store in the header.
        header (Dict[str, str]): The existing response header dictionary.

    Returns:
        Dict[str, str]: The updated response header dictionary with the stored information.
    '''
    mapping = {
        'lang': 'Accept-Language',
        'encoding': 'Accept-Encoding',
        'cache': 'Cache-Control'
    }
    
    if info_type not in mapping:
        return header
    
    key = mapping[info_type]
    
    # Validate info_content based on info_type
    if info_type == 'lang':
        # Validate language tag: e.g., 'en', 'en-US', 'fr-FR'
        if not re.fullmatch(r'^[a-z]{2,3}(-[A-Z]{2,3})?$', info_content):
            return header
    elif info_type == 'encoding':
        # Check against allowed encodings
        allowed_encodings = {'gzip', 'deflate', 'br', 'identity'}
        if info_content not in allowed_encodings:
            return header
    elif info_type == 'cache':
        # Check for valid Cache-Control directives
        allowed_cache_directives = {'no-cache', 'max-age', 'public', 'private', 'must-revalidate'}
        if not any(directive in info_content for directive in allowed_cache_directives):
            return header
    
    try:
        new_header = header.copy()
        new_header[key] = info_content
        return new_header
    except Exception:
        return header