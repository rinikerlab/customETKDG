def get_data_file_path(relative_path):
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``openff/toolkit/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).
    """
    import os

    from pkg_resources import resource_filename

    fn = resource_filename("custom_etkdg", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install"
        )

    return fn


"""
https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish

"""
#XXX but does not actual stop EmbedMultipleConfs after timeout, this approach does not work right now

import os
import errno
from functools import wraps
import signal #XXX only UNIX platforms

class TimeoutError(Exception):
    pass

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator