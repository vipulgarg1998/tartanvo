from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection ='3d')

poses = np.load('results/unity_tartanvo_1914.npy')

plt.plot(poses[:,0], poses[:,1])
# ax.plot3D(poses[:,0], poses[:,1], poses[:,2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

plt.show()