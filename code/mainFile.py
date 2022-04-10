
# -*- coding: utf-8 -*-
import os
import scipy.io as sio
import numpy as np
import cProfile
import utils.viewer_util as viewer
import utils.loc_utils as loc
import build3D.Reconstruction3D as buildTool
import time



if __name__ == '__main__':

    path = r'E:\20220330\BrainA4\640\Rebuild1'

    cProfile.run('main()')
    #定位
    start = time.time()
    loc.sequence_Loc_multiProcess(path)
    end = time.time()
    print("时间：")
    print(end - start)
    #
    #重建
    tool = buildTool.RebuildTool(path, pixelsize=162.5, camera1_zrange=[-3,3], camera2_zrange=[-3,3])
    tool.rebuild_singal3D(filterType=2)
    tool.camera_Rebuild(filterType=2, duplicate_remove=1, radius=1.5)
    tool.registration_camera(duplicate_remove_single=0, r_single=12, duplicate_remove_two=1, r_two=15)
    #
    end = time.time()
    print("时间：")
    print(end - start)

    viewerTool = viewer.viewer(os.path.join(path, 'locFile\\Panorama_Points.mat'))
    viewerTool.viewer_2D()
    viewerTool.viewer_3D()
    viewerTool = viewer.viewer(os.path.join(path, 'locFile\\camera1_rebuild.mat'))
    viewerTool.viewer_2D()
    viewerTool.viewer_3D()
    viewerTool = viewer.viewer(os.path.join(path, 'locFile\\camera2_rebuild.mat'))
    viewerTool.viewer_2D()
    viewerTool.viewer_3D()


    # 筛选结果并保存
    x_range = [0,15000]
    y_range = [0,15000]
    z_range = [1,2500]
    PointCloud3D = sio.loadmat(os.path.join(path, 'rebuild3D.mat'))
    PointCloud3D = PointCloud3D['point3D']
    part = PointCloud3D[:, np.where((PointCloud3D[0,:] < x_range[1]))]
    part = np.squeeze(part)
    part = part[:, np.where((part[0,:] > x_range[0]))]
    part = np.squeeze(part)
    part = part[:, np.where((part[1,:] < y_range[1]))]
    part = np.squeeze(part)
    part = part[:, np.where((part[1,:] > y_range[0]))]
    part = np.squeeze(part)
    part = part[:, np.where((part[2,:] < z_range[1]))]
    part = np.squeeze(part)
    part = part[:, np.where((part[2,:] > z_range[0]))]
    part = np.squeeze(part)
    sio.savemat(os.path.join(path, 'rebuild3D_part .mat'), {'point3D':part})
    viewerTool = viewer.viewer(part)
    viewerTool.viewer_2D()
    viewerTool.viewer_3D()
