# Utility functions

def get_oserror_dir(e: OSError) -> str:
    """Return non-existent directory name raised within OSError. """    
    return str(e).split(': ')[1].strip("''")