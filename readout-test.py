import matplotlib
matplotlib.use("QtAgg")

import lightkurve as lk

search_result = lk.search_targetpixelfile("TOI-2046")
pixelfile = search_result[0].download()

pixelfile.animate()