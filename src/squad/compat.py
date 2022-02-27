import sys


try:
    import pybullet

    HAS_PYBULLET = True
    del pybullet
except ImportError:
    HAS_PYBULLET = False


IS_64 = sys.maxsize > 2 ** 32
