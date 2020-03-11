import numpy as np
import os

root = '/data/all/jianrenw/kitti/tracking/training/velodyne/0000/'
output_path = '/home/ajangid/HPLFlowNet/dataset_processed/KITTI_MOT/000000/'
i = 0
f1 = os.path.join(root,"{:06d}.bin".format(i))
f2 = os.path.join(root,"{:06d}.bin".format(i+1))

pc1 = np.fromfile(f1, dtype=np.float32).reshape(-1, 4)
pc2 = np.fromfile(f2, dtype=np.float32).reshape(-1, 4)
pc1 = pc1[:,:-1] # ignoring intensity
pc2 = pc2[:,:-1]

out1 = os.path.join(output_path,'pc1.npy')
out2 = os.path.join(output_path,'pc2.npy')

np.save(out1, pc1)
np.save(out2, pc2)
print("data written")
# print(pc1[:4])


