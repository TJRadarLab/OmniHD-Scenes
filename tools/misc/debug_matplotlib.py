import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')
ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
plt.show()
