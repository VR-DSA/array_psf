import numpy as np, matplotlib.pyplot as plt
import sys

d = np.loadtxt(sys.argv[1]).reshape((1024,1024))
plt.imshow(d)
plt.show()
