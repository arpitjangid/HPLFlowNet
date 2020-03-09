# Usage: python xxx.py path_to_pc1/pc2/output/epe3d/path_list [pc2]

import numpy as np
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle
import open3d as o3d

SCALE_FACTOR = 0.05
MODE = 'sphere'
DRAW_LINE = False #True
DRAW_PRED_FLOW = False #True # arpit
# DRAW_GT_FLOW = False # one of these two can be True at a time
VIS_MAYAVI = False

if '-h' in ' '.join(sys.argv):
	print('Usage: python3 visu_new.py VISU_PATH')
	sys.exit(0)

visu_path = sys.argv[1]
# visu_path = "./visu_ours_KITTI_8192_35m/"
# all_epe3d = np.load(osp.join(visu_path, 'epe3d_per_frame.npy'))

path_list = None
if osp.exists(osp.join(visu_path, 'sample_path_list.pickle')):
	with open(osp.join(visu_path, 'sample_path_list.pickle'), 'rb') as fd:
		path_list = pickle.load(fd)
		# print("path_list", path_list)
		
for index in range(5): #len(path_list)):
	# if index!=14:
	# 	continue
	pc1 = np.load(osp.join(visu_path, 'pc1_'+str(index)+'.npy')).squeeze()
	pc2 = np.load(osp.join(visu_path, 'pc2_'+str(index)+'.npy')).squeeze()
	# sf = np.load(osp.join(visu_path,  'sf_'+str(index)+'.npy')).squeeze()
	output = np.load(osp.join(visu_path, 'output_'+str(index)+'.npy')).squeeze()
	
	if pc1.shape[1] != 3:
		pc1 = pc1.T
		pc2 = pc2.T
		# sf = sf.T
		output = output.T
	
	## for nuscenes, partial point cloud visualization
	if False:
		num_points = 8000
		factor = 1
		pc1 = pc1[:num_points,:] * factor
		pc2 = pc2[:num_points,:] * factor
		# sf = sf[:num_points,:] * factor
		output = output[:num_points,:] * factor

	# print("pc1 min max", np.min(pc1,axis=0), np.max(pc1, axis=0))
	# gt = pc1 + sf
	pred = pc1 + output
	
	# print('pc1, pc2, gt, pred', pc1.shape, pc2.shape, gt.shape, pred.shape)

	if(VIS_MAYAVI):

		# fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
		fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=(0,0,0), engine=None, size=(1600, 1000))
		
		if True: #len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
			mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
		
		if len(sys.argv) >= 4 and sys.argv[3] == 'pc2':
				mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0,1,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # cyan

		# mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # red
		# mlab.points3d(pred[:, 0], pred[:,1], pred[:,2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # green
		
		# DRAW LINE
		if DRAW_LINE:
			N = 2
			x = list()
			y = list()
			z = list()
			connections = list()

			inner_index = 0
			for i in range(output.shape[0]):
				if DRAW_PRED_FLOW:
					x.append(pc1[i, 0])
					x.append(pred[i, 0])
					y.append(pc1[i, 1])
					y.append(pred[i, 1])
					z.append(pc1[i, 2])
					z.append(pred[i, 2])

				connections.append(np.vstack(
					[np.arange(inner_index,   inner_index + N - 1.5),
					np.arange(inner_index + 1,inner_index + N - 0.5)]
				).T)
				inner_index += N

			x = np.hstack(x)
			y = np.hstack(y)
			z = np.hstack(z)

			connections = np.vstack(connections)

			src = mlab.pipeline.scalar_scatter(x, y, z)

			src.mlab_source.dataset.lines = connections
			src.update()
			
			lines= mlab.pipeline.tube(src, tube_radius=0.005, tube_sides=6)
			mlab.pipeline.surface(lines, line_width=2, opacity=.4, color=(1,1,0))
		# DRAW LINE END

		mlab.view(90, # azimuth
				150, # elevation
				50, # distance
				[0, -1.4, 18], # focalpoint
				roll=0)

		mlab.orientation_axes()

		mlab.show()
	else:
		point_clouds = []
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(pc1)
		pcd1.paint_uniform_color((0.0,0.0,1.0))
		point_clouds.append(pcd1)

		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(pc2)
		pcd2.paint_uniform_color((0.0,1.0,0.0))
		point_clouds.append(pcd2)

		# pcd_pred = o3d.geometry.PointCloud()
		# pcd_pred.points = o3d.utility.Vector3dVector(pred)
		# pcd_pred.paint_uniform_color((1.0,0.0,0.0))
		# point_clouds.append(pcd_pred)

		o3d.visualization.draw_geometries(point_clouds)



	# epe3d = all_epe3d[index]
	# print(epe3d)
	path = path_list[index]
	# print(path, epe3d)	
	
	
	