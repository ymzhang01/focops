import sys

def println(*args):
    """
    Print and flush
    """
    print(*args)
    sys.stdout.flush()