import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")

import lightkurve as lk

for i in range(10):
    pixelfile = lk.search_targetpixelfile("TrES-3 b")[i].download()
    lc = pixelfile.to_lightcurve(method="pld").remove_outliers().flatten()
    period = lc.to_periodogram("bls").period_at_max_power
    lc.fold(period).scatter();
    plt.show()
    
