import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
import scipy.io as sio


class viewer(object):


    def __init__(self, file):
        self.file = file


    def viewer_3D(self):
        '''
        :argument
        Visualize 3D points in 3D room
        :parameter
        Input:3D points ,type:3xN numpy array
        '''
        PointCloud3D = sio.loadmat(self.file)
        PointCloud3D = PointCloud3D['Dao3D_Result']
        PointCloud3D = PointCloud3D[0:3, :]
        pcd = o3d.geometry.PointCloud()#传入3d点云
        pcd.points = o3d.utility.Vector3dVector(PointCloud3D.transpose().tolist())
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        render_option: o3d.visualization.RenderOption = vis.get_render_option()
        render_option.background_color = np.array([0, 0, 0])
        render_option.point_size = 2.0
        vis.add_geometry(pcd)
        vis.run()


    def viewer_2D(self):
        '''
        :argument
        Visualize 3D points with 2D-depth plot
        :parameter
        Input:3D points ,type:3xN numpy array
        '''
        PointCloud3D = sio.loadmat(self.file)
        PointCloud3D = PointCloud3D['Dao3D_Result']
        plt.figure(dpi=300, figsize=(24,8))
        plt.rcParams['axes.facecolor'] = 'black'#设置背景色
        plt.scatter(PointCloud3D[0, :], PointCloud3D[1, :], c=PointCloud3D[2, :], marker="o", s=0.5, cmap='jet')
        plt.xlabel('X/μm', fontdict={'size':20})
        plt.ylabel('Y/μm', fontdict={'size':20})
        plt.xlim(0, 700)
        plt.ylim(0, 600)
        plt.tick_params(labelsize=16)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
        cb.set_label('Z/μm', fontdict={'size':20})
        plt.axis('equal')

        plt.ion()#让程序到达show后能够往下执行
        plt.show()