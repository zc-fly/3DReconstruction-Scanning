import numpy as np
import os
import scipy.io as sio
import copy
#import open3d as o3d
from glob import glob
from itertools import islice
import CalibrationAndRegistration.registrationTWOcamera as regTool
import afterProcess.afterProcess as myFilter


class RebuildTool(object):

    def __init__(self, path, pixelsize=182, camera1_zrange=[-4,4], camera2_zrange=[-4,4]):
        self.path = path
        self.pixelsize = pixelsize
        self.camera1_HW = [0, 0]
        self.camera2_HW = [0, 0]
        self.camera1_zrange = camera1_zrange
        self.camera2_zrange = camera2_zrange


    def __frameInfoReader(self):
        """
        读取frameInfo信息，分别获取两个相机的所有文件帧数
        :return:
        """
        frameInfo_camera1 = []
        frameInfo_camera2 = []
        frames_camera1 = [0]
        frames_camera2 = [0]
        with open(self.path + "\\locFile\\frameInfo.txt", 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                    pass
                if lines.find('camera1') != -1:
                    frameInfo_camera1.append(lines)
                if lines.find('camera2') != -1:
                    frameInfo_camera2.append(lines)
                pass

        frameInfo_camera1.sort(key=lambda x: int(x.split('_part_')[1].split('.tif')[0]))
        frameInfo_camera2.sort(key=lambda x: int(x.split('_part_')[1].split('.tif')[0]))
        for c1 in frameInfo_camera1:
            p_tmp, e_tmp = [i for i in c1.split('.tif')]
            frames_camera1.append(int(e_tmp))
        for c2 in frameInfo_camera2:
            p_tmp, e_tmp = [i for i in c2.split('.tif')]
            frames_camera2.append(int(e_tmp))
        pass
        return [frames_camera1, frames_camera2]


    def __StageStepINFO(self):
        """
        :argument 读取位移台移动步长信息
        :return:
        """
        displaceFile = glob(f'{self.path}/*camera*.txt', recursive=True)
        displaceInfo = []
        with open(displaceFile[0], "r") as f:
            for line in f.readlines():
                data = line.split()
                displaceInfo.append(np.array(data))
        camera1_HW = displaceInfo[13]
        camera2_HW = displaceInfo[15]
        displaceInfo = displaceInfo[18:]
        displaceInfo = np.array(displaceInfo)
        displaceInfo = displaceInfo[:, 1:4]
        camera1_HW = np.array(camera1_HW)
        camera2_HW = np.array(camera2_HW)
        self.camera1_HW = camera1_HW[2:4]
        self.camera2_HW = camera2_HW[2:4]

        return displaceInfo


    def rebuild_singal3D(self, filterType=2, PSF_range=[1.1,8,1.1,8]):
        """
        :argument: 根据标定曲线和sigmax,sigmay，重建每帧图像上点的三维坐标（不对扫描结果进行累加，每帧图像空间坐标系独立）
        filter = 0:不对重建结果进行筛选
                 1:对重建获得的z轴深度进行筛选
                 2:对点的z轴深度和PSF大小进行筛选
        PSF_range = [sigmax_min, sigmax_max, sigmay_min, sigmay_max]
        :return: 返回每个定位文件3D重建后的结果文件
        """
        path = self.path
        path_register = path + '\\registerFile'
        curve1 = glob(f'{path_register}/camera1*.mat', recursive=True)
        curve2 = glob(f'{path_register}/camera2*.mat', recursive=True)
        if len(curve1) == 0 or len(curve2) == 0:
            print("check your registerFile!")
            exit(0)

        path_loc = path + '\\locFile'
        locFile_1 = glob(f'{path_loc}/*MultiFov_camera1*.mat', recursive=True)
        locFile_2 = glob(f'{path_loc}/*MultiFov_camera2*.mat', recursive=True)

        calibs1 = sio.loadmat(curve1[0])
        calibs2 = sio.loadmat(curve2[0])

        for p, data1 in enumerate(locFile_1):
            locPoint_temp_1 = sio.loadmat(data1)
            locPoint_temp_1 = locPoint_temp_1['Dao3D_Result']
            sigmaxy = locPoint_temp_1[3] ** 2 - locPoint_temp_1[4] ** 2
            locPoint_temp_1[2] = calibs1['p1'] * sigmaxy ** 4 + calibs1['p2'] * sigmaxy ** 3 + calibs1[
                'p3'] * sigmaxy ** 2 + calibs1['p4'] * sigmaxy
            if filterType == 0:
                sio.savemat(data1.split('MultiFov_')[0] + '3D_Nonefilter_' + data1.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_1})
            if filterType == 1:
                locPoint_temp_1 = locPoint_temp_1[:, (locPoint_temp_1[2] > self.camera1_zrange[0]) & (locPoint_temp_1[2] < self.camera1_zrange[1])]
                sio.savemat(data1.split('MultiFov_')[0] + '3D_Zfilter_' + data1.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_1})
            if filterType == 2:
                locPoint_temp_1 = locPoint_temp_1[:, (locPoint_temp_1[2] > self.camera1_zrange[0]) & (locPoint_temp_1[2] < self.camera1_zrange[1])]
                locPoint_temp_1 = locPoint_temp_1[:, (locPoint_temp_1[3] > PSF_range[0]) & (locPoint_temp_1[3] < PSF_range[1])]
                locPoint_temp_1 = locPoint_temp_1[:, (locPoint_temp_1[4] > PSF_range[2]) & (locPoint_temp_1[4] < PSF_range[3])]
                sio.savemat(data1.split('MultiFov_')[0] + '3D_PSFfilter_'+ data1.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_1})

        for q, data2 in enumerate(locFile_2):
            locPoint_temp_2 = sio.loadmat(data2)
            locPoint_temp_2 = locPoint_temp_2['Dao3D_Result']
            sigmaxy = locPoint_temp_2[3] ** 2 - locPoint_temp_2[4] ** 2
            locPoint_temp_2[2] = calibs2['p1'] * sigmaxy ** 4 + calibs2['p2'] * sigmaxy ** 3 + calibs2[
                'p3'] * sigmaxy ** 2 + calibs2['p4'] * sigmaxy
            if filterType == 0:
                sio.savemat(data2.split('MultiFov_')[0] + '3D_Nonefilter_' + data2.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_2})
            if filterType == 1:
                locPoint_temp_2 = locPoint_temp_2[:, (locPoint_temp_2[2] > self.camera2_zrange[0]) & (locPoint_temp_2[2] < self.camera2_zrange[1])]
                sio.savemat(data2.split('MultiFov_')[0] + '3D_Zfilter_' + data2.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_2})
            if filterType == 2:
                locPoint_temp_2 = locPoint_temp_2[:, (locPoint_temp_2[2] > self.camera2_zrange[0]) & (locPoint_temp_2[2] < self.camera2_zrange[1])]
                locPoint_temp_2 = locPoint_temp_2[:, (locPoint_temp_2[3] > PSF_range[0]) & (locPoint_temp_2[3] < PSF_range[1])]
                locPoint_temp_2 = locPoint_temp_2[:, (locPoint_temp_2[4] > PSF_range[2]) & (locPoint_temp_2[4] < PSF_range[3])]
                sio.savemat(data2.split('MultiFov_')[0] + '3D_PSFfilter_'+ data2.split('MultiFov_')[1], {'Dao3D_Result': locPoint_temp_2})
        print('singal File rebuild Complete!')


    def camera_Rebuild(self, filterType=2, duplicate_remove=0, radius=2):
        """
        读取连续扫描的图像文件进行重建
        filterType = 0:对未过滤的3D点进行重建（重建后缀为_3D_Nonefilter.mat的文件）
                   = 1:对进行z轴深度滤波的3D点进行重建（重建后缀为_3D_Zfilter.mat的文件）
                   = 2:对进行PSF大小滤波的3D点进行重建（重建后缀为_3D_PSFfilter.mat的文件）
        duplicate_remove = 0,1 是否进行半径滤波
        radius: 半径滤波半径值
        :return:返回两个相机单独拼接重建的3D结果文件
        """
        path = self.path

        path_loc = path + '\\locFile'
        if filterType==0:
            locFile_1 = glob(f'{path_loc}/*Nonefilter*camera1*.mat', recursive=True)
            locFile_2 = glob(f'{path_loc}/*Nonefilter*camera2*.mat', recursive=True)

        elif filterType==1:
            locFile_1 = glob(f'{path_loc}/*Zfilter*camera1*.mat', recursive=True)
            locFile_2 = glob(f'{path_loc}/*Zfilter*camera2*.mat', recursive=True)

        elif filterType==2:
            locFile_1 = glob(f'{path_loc}/*PSFfilter*camera1*.mat', recursive=True)
            locFile_2 = glob(f'{path_loc}/*PSFfilter*camera2*.mat', recursive=True)

        else:
            print('Appoint correct rebuild type!')

        locFile_1.sort(key=lambda x: int(x.split('_part_')[1].split('.mat')[0]))
        locFile_2.sort(key=lambda x: int(x.split('_part_')[1].split('.mat')[0]))

        frameInfo = self.__frameInfoReader()

        locPoint_1 = np.empty(shape=(13, 0))
        locPoint_2 = np.empty(shape=(13, 0))

        for p, data1 in enumerate(locFile_1):
            locPoint_temp_1 = sio.loadmat(data1)
            locPoint_temp_1 = locPoint_temp_1['Dao3D_Result']
            locPoint_temp_1[12,:] = locPoint_temp_1[12,:] + sum(islice(frameInfo[0], p+1))
            locPoint_1 = np.hstack((locPoint_1, locPoint_temp_1))

        for q, data2 in enumerate(locFile_2):
            locPoint_temp_2 = sio.loadmat(data2)
            locPoint_temp_2 = locPoint_temp_2['Dao3D_Result']
            locPoint_temp_2[12, :] = locPoint_temp_2[12, :] + sum(islice(frameInfo[1], q+1))
            locPoint_2 = np.hstack((locPoint_2, locPoint_temp_2))

        displaceInfo = self.__StageStepINFO()
        locPoint_1[0, :] = locPoint_1[0, :] - int(self.camera1_HW[0]) / 2
        locPoint_1[1, :] = locPoint_1[1, :] - int(self.camera1_HW[1]) / 2
        locPoint_2[0, :] = locPoint_2[0, :] - int(self.camera2_HW[0]) / 2
        locPoint_2[1, :] = locPoint_2[1, :] - int(self.camera2_HW[1]) / 2
        transform = regTool.transform_of_pointclouds(self.path)
        for i in range(1, int(displaceInfo.shape[0])+1):
            
            index1 = locPoint_1[12, :] == i
            index2 = locPoint_2[12, :] == i
            locPoint_1[0, index1] = locPoint_1[0, index1] * self.pixelsize / 1000 - 1000 * float(displaceInfo[i - 1, 1])
            temp = locPoint_1[1, index1].copy()
            locPoint_1[1, index1] = temp * self.pixelsize / (1000 * (2 ** 0.5)) - locPoint_1[2, index1] /\
                                    (2 ** 0.5) + 1000 * float(displaceInfo[i - 1, 0])
            locPoint_1[2, index1] = temp * self.pixelsize / (1000 * (2 ** 0.5)) + locPoint_1[2, index1] / \
                                    (2 ** 0.5) + 1000 * float(displaceInfo[i - 1, 2])
            locPoint_1[0:3, index1] = np.dot(transform[0:3, 0:3], locPoint_1[0:3, index1])

            locPoint_2[0, index2] = locPoint_2[0, index2] * self.pixelsize / 1000 - 1000 * float(displaceInfo[i - 1, 1])
            temp = locPoint_2[1, index2].copy()
            locPoint_2[1, index2] = temp * self.pixelsize / (1000 * (2 ** 0.5)) - locPoint_2[2, index2] / \
                                    (2 ** 0.5) + 1000 * float(displaceInfo[i - 1, 0])
            locPoint_2[2, index2] = temp * self.pixelsize / (1000 * (2 ** 0.5)) + locPoint_2[2, index2] / \
                                    (2 ** 0.5) + 1000 * float(displaceInfo[i - 1, 2])

        if duplicate_remove==1:
            labels1 = myFilter.radius_filter(locPoint_1, radius)
            locPoint_1 = locPoint_1[:, labels1 == -1]
            labels2 = myFilter.radius_filter(locPoint_2, radius)
            locPoint_2 = locPoint_2[:, labels2 == -1]

        sio.savemat(os.path.join(path_loc, 'camera1_rebuild.mat'), {'Dao3D_Result': locPoint_1})
        sio.savemat(os.path.join(path_loc, 'camera2_rebuild.mat'), {'Dao3D_Result': locPoint_2})
        print('Two Camera rebuild Complete!')


    def registration_camera(self, duplicate_remove_single=0, r_single=15, duplicate_remove_two=0, r_two=15):
        """
        :argument 加载两相机单独重建结果，进行配准，并进行连续帧重复点去除
        :return: 最终重建结果文件输出
        """
        path_loc = os.path.join(self.path, 'locFile')
        #加载两相机点云
        camera1_path = glob(f'{path_loc}/camera1_rebuild.mat', recursive=True)
        camera2_path = glob(f'{path_loc}/camera2_rebuild.mat', recursive=True)
        camera1 = sio.loadmat(camera1_path[0])
        camera1 = camera1['Dao3D_Result']
        camera2 = sio.loadmat(camera2_path[0])
        camera2 = camera2['Dao3D_Result']

        if duplicate_remove_single == 1:
            displaceInfo = self.__StageStepINFO()
            camera1 = myFilter.duplication_remove(camera1, displaceInfo, PixelSize=self.pixelsize, r=r_single)
        if duplicate_remove_single == 1:
            displaceInfo = self.__StageStepINFO()
            camera2 = myFilter.duplication_remove(camera2, displaceInfo, PixelSize=self.pixelsize,r=r_single)

        transform = regTool.transform_of_pointclouds(self.path)
        camera1[0:3,:] = camera1[0:3, :] + transform[0:3, 3].reshape((3,-1))

        Panorama_Points = np.hstack((camera1, camera2))
        if duplicate_remove_two==1:
            displaceInfo = self.__StageStepINFO()
            Panorama_Points = myFilter.duplication_remove(Panorama_Points, displaceInfo, PixelSize=self.pixelsize, r=r_two)

        sio.savemat(os.path.join(path_loc, 'Panorama_Points.mat'), {'Dao3D_Result': Panorama_Points})

        print('Rebuild Complete!')













