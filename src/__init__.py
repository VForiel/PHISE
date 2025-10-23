"""Module generated docstring."""
try:
	import matplotlib.pyplot as plt
	try:
		plt.rcParams['image.origin'] = 'lower'
	except Exception:
		pass
except Exception:
	plt = None