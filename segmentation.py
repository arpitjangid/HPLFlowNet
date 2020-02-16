import numpy as np
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.cluster import OPTICS
import open3d as o3d

SCALE_FACTOR = 0.05
MODE = 'sphere'

def dbscan(features, min_samples=50, eps = 1, if_optics=False):
	if(if_optics):
		db = OPTICS(min_samples=min_samples).fit(features)
	else:
		db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
	
	labels = db.labels_
	unique_labels = set(labels)
	# print("unique labels",len(unique_labels)) # including noise
	label_indxs = []
	for la in unique_labels:
		if la==-1: # noise
			noise_ids = np.where(labels == la)[0]
			continue
		ids = np.where(labels == la)[0]
		label_indxs.append(ids)
	label_indxs.append(noise_ids) # noise points in the end
	# print("label_indxs",len(label_indxs))
	return label_indxs #, noise_ids
		
def kmeans(features, n_clusters):
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
	labels = kmeans.labels_
	print("labels",labels.shape)
	# label_indxs = {}
	label_indxs = []
	for i in range(n_clusters):
		ids = np.where(labels==i)[0]
		# print("ids",ids)
		# label_indxs[i] = np.array(ids)
		label_indxs.append(ids)
	# print("label_indxs", label_indxs)
	
	return label_indxs

def display_clust_mayavi(clusters, coords):
	fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
	# # colorlist = [(1,0,0),(0,0,1), (0,1,1)]
	# num_cl = len(clusters)
	# colorlist = [(0.4*x,0.8*x,1) for x in np.arange(1/num_cl,1.0001,1/num_cl)]
	for i, cluster in enumerate(clusters):
		fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
		mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
		cluster_coords = coords[cluster]
		# print("i, len colorlist",i, len(colorlist))
		# print("color",colorlist[i])

		mlab.points3d(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE)
		# mlab.points3d(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2], color=colorlist[i], scale_factor=SCALE_FACTOR, figure=fig, mode=MODE)
		# mlab.show()

	mlab.show()
	pass

def get_lines(bbox):
	inds = np.array([[0, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[1, 1, 0],
		[0, 0, 1],
		[1, 0, 1],
		[0, 1, 1],
		[1, 1, 1]])
	inds[np.where(inds==1)] += 2 # to shift to max coordinate indices
	inds += np.array([0,1,2])
	points = bbox[inds]
	# print("bbox", bbox)
	# print("points",points)
	# exit()
	lines = np.array([
		[0, 1],
		[0, 2],
		[1, 3],
		[2, 3],
		[4, 5],
		[4, 6],
		[5, 7],
		[6, 7],
		[0, 4],
		[1, 5],
		[2, 6],
		[3, 7],
		])
	return points, lines

def display_clust_o3d(clusters, coords, bboxes):
	barColors = [(240,163,255),(0,117,220),(153,63,0),
	(76,0,92),(25,25,25),(0,92,49),
	(43,206,72),(255,204,153),(128,128,128),
	(148,255,181),(143,124,0),(157,204,0),(194,0,136),
	(0,51,128),(255,164,5),(255,168,187),(66,102,0),
	(255,0,16),(94,241,242),(0,153,143),(224,255,102),
	(116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5)]
	noiseColor = (np.array([(0,0,0)])).astype(np.float_)
	# noiseColor = (np.array([(0,1,0)])).astype(np.float_)

	barColors = (np.array(barColors)/255).astype(np.float_)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(coords)
	num_points = len(coords)
	color_vec = np.zeros((num_points,3),dtype=np.float_)
	for i, indxs in enumerate(clusters):
		if i==len(clusters)-1: # noise points
			print("noise points")
			color_vec[indxs] = noiseColor
		else:
		# cluster_coords = coords[cluster]
		# pcd.points = o3d.utility.Vector3dVector(cluster_coords)
			color_vec[indxs] = barColors[i%len(barColors)]
		
	pcd.colors = o3d.utility.Vector3dVector(color_vec)
	# o3d.io.write_point_cloud("tmp.ply", pcd)
	# pcd_load = o3d.io.read_point_cloud("tmp.ply")
	# o3d.visualization.draw_geometries([pcd])

	
	#drawing object bboxes
	points = []
	lines = []
	for i in range(len(bboxes)-1): # ignoring last bbox (represents noise in current segmentation)
		p, l = get_lines(bboxes[i]) # expecting p and l to be np arrays
		points.append(p)
		lines.append(l + i*8)
	points = np.vstack(points)
	lines = np.vstack(lines)
	line_set = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(points),
		lines=o3d.utility.Vector2iVector(lines),
	)
	colors = [[1, 0, 0] for i in range(8*len(bboxes))]
	line_set.colors = o3d.utility.Vector3dVector(colors)
	o3d.visualization.draw_geometries([pcd, line_set])
	
		
def get_bboxes(clusters, coords):
	out = []
	for i,indxs in enumerate(clusters):
		cluster_coords = coords[indxs]
		xyz_min = np.min(cluster_coords,axis=0)
		xyz_max = np.max(cluster_coords,axis=0)
		out.append(np.concatenate((xyz_min,xyz_max)))
	return np.vstack(out)

if __name__ == "__main__":
	# visu_path = sys.argv[1]
	visu_path = "./visu_ours_KITTI_8192_35m/"
	all_epe3d = np.load(osp.join(visu_path, 'epe3d_per_frame.npy'))

	path_list = None
	if osp.exists(osp.join(visu_path, 'sample_path_list.pickle')):
		with open(osp.join(visu_path, 'sample_path_list.pickle'), 'rb') as fd:
			path_list = pickle.load(fd)
			# print("path_list", path_list)

	flow_weight = 10 #10 #20
	for index in range(10):#len(path_list)):
		print ("results for index:", index)
		# if index != 4:
		# 	continue
		pc1 = np.load(osp.join(visu_path, 'pc1_'+str(index)+'.npy')).squeeze()
		sf = np.load(osp.join(visu_path,  'sf_'+str(index)+'.npy')).squeeze() # to check results with gt sceneflow
		output = np.load(osp.join(visu_path, 'output_'+str(index)+'.npy')).squeeze()
		print("path", path_list[index])
		if pc1.shape[1] != 3:
			pc1 = pc1.T
			sf = sf.T
			output = output.T

		feat_output = np.hstack((pc1,flow_weight*output))
		feat_gt = np.hstack((pc1,sf))
		print('feat_output, feat_gt',feat_output.shape, feat_gt.shape)
		# n_clusters = 5
		min_samples = 50
		eps = 1
		clusters_out = dbscan(feat_output,min_samples=min_samples, eps = eps)
		print("num clusters pred sceneflow",len(clusters_out))
		bboxes = get_bboxes(clusters_out, pc1)
		display_clust_o3d(clusters_out, pc1, bboxes)
		# print('bboxes', bboxes.shape)
		
		clusters_gt = dbscan(feat_gt, min_samples=min_samples, eps = eps)
		bboxes = get_bboxes(clusters_gt, pc1)
		print("num clusters gt",len(clusters_gt))
		display_clust_o3d(clusters_gt, pc1, bboxes)

