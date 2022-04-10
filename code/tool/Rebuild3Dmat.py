# -*- coding: utf-8 -*-
from libtiff import TIFFfile
from glob import glob
import scipy.io as sio
import numpy as np
import os
import utils.loc_utils as loc


class RebuildTool(object):

    def __init__(self, path, pixelsize=182, camera1_zrange=[-4,4]):
        self.path = path
        self.pixelsize = pixelsize
        self.camera1_zrange = camera1_zrange

    def __file_filter(self,f):
        if f[-4:] in ['.mat']:
            return True
        else:
            return False

    def Rebuild(self):

        #文件路径筛选
        path = self.path
        path_register = path + '\\registerFile'#配准与标定路径
        curve1 = glob(f'{path_register}/camera1*.mat', recursive=True)
        if len(curve1)==0:
            print("check your registerFile!")
            exit(0)

        path_loc = path + '\\locFile'  # 定位表路径,分别筛选出两个相机定位表
        locFile_1 = glob(f'{path_loc}/myfile*.mat', recursive=True)
        File_img1 = glob(f'{path}/myfile*.tif', recursive=True)

        locFile_1.sort(key=lambda x:int(x.split('myfile_')[1].split('.mat')[0]))
        File_img1.sort(key=lambda x: int(x.split('myfile_')[1].split('.tif')[0]))

        #加载标定曲线、双相机相对位置标定、位移台位移信息、定位结果
        calibs1 = sio.loadmat(curve1[0])  # 加载标定曲线

        locPoint_1 = np.empty(shape=(13, 0))#x, y, z, xsigma, ysigma, height, Sum, background, category, error,iterations, significance, frame_t

        startFrames1 = 0
        for p, data1 in enumerate(locFile_1):#将多个stack图像文件定位结果进行统一，获得帧数正确的定位表
            locPoint_temp_1 = sio.loadmat(data1)
            locPoint_temp_1 = locPoint_temp_1['Dao3D_Result']
            locPoint_temp_1[12,:] = locPoint_temp_1[12,:] + startFrames1
            locPoint_1 = np.hstack((locPoint_1, locPoint_temp_1))
            tif = TIFFfile(File_img1[p])#获取stack帧数
            stacksize = tif.get_depth()
            tif.close()
            startFrames1 = stacksize + startFrames1

        start_frame = locPoint_1[12, 0]
        end_frame = locPoint_1[12, -1]

        Dao3D_Result1 = np.empty(shape=(13, 0))
        for i in range(int(start_frame), int(end_frame)): # 遍历两相机相同帧图像
            img1 = locPoint_1[:, locPoint_1[12,:]==i]
            for j in range(0, img1.shape[1]):#遍历第1个相机第i帧图像上的所有点
                sigmaxy = img1[3][j] ** 2 - img1[4][j] ** 2
                img1[2][j] = calibs1['p1'] * sigmaxy ** 4 + calibs1['p2'] * sigmaxy ** 3 + calibs1['p3'] * sigmaxy **2 + calibs1['p4'] * sigmaxy
                if img1[2][j] < self.camera1_zrange[0] or img1[2][j] > self.camera1_zrange[1]: continue
                temp_img1 = img1[:, j].reshape((13,-1))
                Dao3D_Result1 = np.hstack((Dao3D_Result1, temp_img1))

        sio.savemat(os.path.join(path, 'Total.mat'), {'Dao3D_Result': Dao3D_Result1})




def rebuild_singal3D(path, filter=0, camera1_zrange=[-4,4], camera2_zrange=[-4,4], PSF_range=[-5, 5, -5, 5]):
    """
    :argument: 根据标定曲线和sigmax,sigmay，重建每帧图像上点的三维坐标（不对扫描结果进行累加，每帧图像空间坐标系独立）
    filter = 0:不对重建结果进行筛选
             1:对重建获得的z轴深度进行筛选
             2:对点的z轴深度和PSF大小进行筛选
    PSF_range = [sigmax_min, sigmax_max, sigmay_min, sigmay_max]
    """
    # 文件路径筛选
    path_register = path + '\\registerFile'  # 配准与标定路径
    curve1 = glob(f'{path_register}/camera1*.mat', recursive=True)

    if len(curve1) == 0:
        print("check your registerFile!")
        exit(0)

    path_loc = path + '\\locFile'  # 定位表路径,分别筛选出两个相机定位表
    locFile = glob(f'{path_loc}/*.mat', recursive=True)

    # 加载标定曲线、双相机相对位置标定、位移台位移信息、定位结果、各个tif帧数信息
    calibs1 = sio.loadmat(curve1[0])  # 加载标定曲线

    for p, data1 in enumerate(locFile):  # 将多个stack图像文件定位结果进行统一，获得帧数正确的定位表
        locPoint_temp_1 = sio.loadmat(data1)
        locPoint_temp_1 = locPoint_temp_1['Dao3D_Result']
        sigmaxy = locPoint_temp_1[3] ** 2 - locPoint_temp_1[4] ** 2
        locPoint_temp_1[2] = calibs1['p1'] * sigmaxy ** 4 + calibs1['p2'] * sigmaxy ** 3 + calibs1[
            'p3'] * sigmaxy ** 2 + calibs1['p4'] * sigmaxy
        if filter == 0:
            sio.savemat(data1.split('.mat')[0] + '_3D_Nonefilter.mat', {'Dao3D_Result': locPoint_temp_1})
        if filter == 1:
            locPoint_temp_1 = locPoint_temp_1[:,
                              locPoint_temp_1[2] > camera1_zrange[0] & locPoint_temp_1[2] < camera1_zrange[1]]
            sio.savemat(data1.split('.mat')[0] + '_3D_Zfilter.mat', {'Dao3D_Result': locPoint_temp_1})
        if filter == 2:
            locPoint_temp_1 = locPoint_temp_1[:,
                              locPoint_temp_1[2] > camera1_zrange[0] & locPoint_temp_1[2] < camera1_zrange[
                                  1]]  # z filter
            locPoint_temp_1 = locPoint_temp_1[:,
                              locPoint_temp_1[3] > camera1_zrange[0] & locPoint_temp_1[3] < camera1_zrange[
                                  1]]  # sigmax filter
            locPoint_temp_1 = locPoint_temp_1[:,
                              locPoint_temp_1[4] > camera1_zrange[2] & locPoint_temp_1[4] < camera1_zrange[
                                  3]]  # sigmay filter
            sio.savemat(data1.split('.mat')[0] + '_3D_PSFfilter.mat', {'Dao3D_Result': locPoint_temp_1})



if __name__ == '__main__':

    path = r'D:\A_CodeFile\matlab\1_MyProject\YQYZ'
    # loc.sequence_Loc_multiProcess(path)
    tool = RebuildTool(path, pixelsize=162.8, camera1_zrange=[-10, 10])
    tool.Rebuild()
    # rebuild_singal3D(path, filter=0, camera1_zrange=[-4, 4], camera2_zrange=[-4, 4], PSF_range=[-50, 50, -50, 50])