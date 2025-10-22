"""Module generated docstring."""
try:
	import matplotlib.pyplot as plt
	try:
		plt.rcParams['image.origin'] = 'lower'
	except Exception:
		# Some mocked matplotlib backends or minimal stubs may not accept assignments
		pass
except Exception:
	plt = None

from .modules import *
from .classes import *