from typing import Dict, Optional
from lxml import etree

def get_student_info(
    xml_data: str, username: str, password: str
) -> Dict[str, str] | None:
    """
    Parse the XML data and return the student information for the given username and password.
    The XML data is expected to have a structure with <students> containing <student> elements
    with <username>, <password>, <name>, <citizenship>, and <age> fields.

    Args:
        xml_data: The XML data containing the student information.
        username: The username of the student to search for.
        password: The password of the student to search for.

    Returns:
        Dict[str, str] | None: A dictionary containing the student information if a matching
        student is found. Otherwise, returns None.
    """
    try:
        # Parse the XML data
        tree = etree.fromstring(xml_data)
    except etree.XMLSyntaxError:
        # Return None if XML parsing fails
        return None

    # Use parameterized XPath to prevent injection attacks
    xpath_query = "//student[username = $username and password = $password]"
    students = tree.xpath(xpath_query, username=username, password=password)

    if not students:
        return None

    student = students[0]
    # Extract the required fields from the student element
    try:
        info = {
            'username': student.find('username').text,
            'password': student.find('password').text,
            'name': student.find('name').text,
            'citizenship': student.find('citizenship').text,
            'age': student.find('age').text,
        }
    except AttributeError:
        # Handle case where expected elements are missing
        return None

    return info
