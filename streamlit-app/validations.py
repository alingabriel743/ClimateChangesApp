import re

def validate_email(email):
    """Validate the email format."""
    pattern = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    return re.match(pattern, email.lower())

def validate_password(password):
    """Ensure the password has a minimum length of 8, contains at least one number, one uppercase letter, and one special character."""
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(pattern, password)

def validate_username(username):
    """Check that the username is alphanumeric and between 4 and 20 characters long."""
    pattern = r'^\w{4,20}$'
    return re.match(pattern, username)