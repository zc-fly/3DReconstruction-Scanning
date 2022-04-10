# -*- coding: utf-8 -*-
"""
:argument registrate TWO point clouds, output its R&T paraments.
output: 变换参数transformation
"""
import numpy as np
import math
import os
import scipy.io as sio


# 双相机配准
def transform_of_pointclouds(path):
    """
    根据输入点云求解变换矩阵
    :param source:
    :param target:
    :param mode:
    :return:
    """

    CamaraCorrect = path + '\\registerFile\\registration.mat'
    try:
        camaraCorrec = sio.loadmat(os.path.join(CamaraCorrect))  # 加载两相机配准值
    except:
        print("check your registration.mat!")
    XYZ = camaraCorrec['resultXYZ']

    angleRotate = 0.001  # 旋转角度
    XYZ[0, 0] = XYZ[0, 0] * 182.5 / 1000 - 0
    XYZ[1, 0] = XYZ[1, 0] * 182.5 / 1000 - 0
    XYZ[2, 0] = XYZ[2, 0] -0

    trans_init = np.asarray([[1, 1, 1, -1.49],#初始化配准参数
                             [1, 1, 1, 14.64],
                             [1, 1, 1, 3.3],
                             [0, 0, 0, 1.0]])

    trans_init[0: 3, 3:4] = -XYZ.reshape(3,-1) #平移变换
    trans_init[0: 3, 0: 3] = np.asarray([[math.cos(angleRotate), -math.sin(angleRotate), 0],#根据角度，计算手动更新旋转矩阵（沿著Z軸旋轉）
                                        [math.sin(angleRotate), math.cos(angleRotate), 0],
                                        [0, 0, 1]])
    # trans_init[0: 3, 0: 3] = np.asarray([[math.cos(angleRotate), 0, math.sin(angleRotate)],#根据角度，计算手动更新旋转矩阵（沿著y軸旋轉）
    #                                      [0, 1, 0],
    #                                     [-math.sin(angleRotate), 0,  math.cos(angleRotate)]])
    # trans_init[0: 3, 0: 3] = np.asarray([[1, 0, 0],
    #                                     [0, math.cos(angleRotate), -math.sin(angleRotate)],#根据角度，计算手动更新旋转矩阵（沿著x軸旋轉）
    #                                     [0, math.sin(angleRotate), math.cos(angleRotate)]])
    return trans_init




    # camera1_xyz = np.array([(10, 20), (10, 10), (10, 10)])
    # camera2_xyz = np.array([(5, 5), (10, 20), (10, 10)])
    # pointCloud3D_1 = camera1_xyz
    # pointCloud3D_1 = pointCloud3D_1.transpose().tolist()
    # pcd_1 = o3d.geometry.PointCloud()  # 传入3d点云
    # pcd_1.points = o3d.utility.Vector3dVector(pointCloud3D_1)
    # pointCloud3D_2 = camera2_xyz
    # pointCloud3D_2 = pointCloud3D_2.transpose().tolist()
    # pcd_2 = o3d.geometry.PointCloud()  # 传入3d点云
    # pcd_2.points = o3d.utility.Vector3dVector(pointCloud3D_2)

    #配准测试
    #P2L method
    # loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    # p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    # reg_p2l = o3d.pipelines.registration.registration_icp(pcd_1, pcd_2,
    #                                                       0.1, trans_init,
    #                                                       p2l)
    # print(reg_p2l.transformation)

    #P2P method
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd_1, pcd_2, 0.02, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print('Total rebuild Complete!')
    # return reg_p2p.transformation


