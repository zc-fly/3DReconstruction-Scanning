# -*- coding: utf-8 -*-
import os
import shutil
import cv2
import numpy as np
from glob import glob
from PIL import Image
import scipy.io as sio
import utils.loc_utils as locTool
from libtiff import TIFFfile, TIFF


def locPaint2img(Filepath, Filename, PSF_range=[1.1,3,1.1,5]):
    """
    :argument: 对.tif文件进行定位，并将定位结果打点在原图上保存（支持单帧/多帧.tif）
    :param Filepath: 文件路径
    :param Filename: 文件名
    :return: 定位打点结果
    """

    Filename = os.path.join(Filepath, Filename)

    #获取图像定位点
    locTool.sequence_Loc_oneFile(Filename)
    portion = os.path.splitext(Filename)  # 将文件名拆成名字和后缀
    locPoint = sio.loadmat((portion[0] +'.mat'))
    locPoint = locPoint['Dao3D_Result']
    locPoint = locPoint[:, (locPoint[3, :] > PSF_range[0]) & (locPoint[3, :] < PSF_range[1])]  # sigmax filter
    locPoint = locPoint[:, (locPoint[4, :] > PSF_range[2]) & (locPoint[4, :] < PSF_range[3])]  # sigmay filter

    #获取图像打点并保存
    try:
        tif = TIFFfile(Filename)  # 获取stack帧数
        stacksize = tif.get_depth()
        tif.close()
    except:
        print("TIF FILE Broken!")

    tif = TIFF.open(Filename, mode='r')

    if stacksize ==1:   #单帧图像文件处理
        img = tif.read_image()
        onePoint = np.array(locPoint, dtype=np.int0) #转换为opencv可读数据类型
        temp_img = np.array(img, dtype=np.float32)  #转换为opencv可读数据类型
        cv2.imshow('img', temp_img)
        for t in range(0, onePoint.shape[1]):   #标记定位结果
            cv2.circle(temp_img, (onePoint[0, t], onePoint[1, t]), radius=1, color=(0,0,255), thickness=1)
        cv2.imwrite((portion[0] + '_Paint.tif'), temp_img)  #保存
        cv2.imshow('img', temp_img)
        tif.close()

    if stacksize > 1:   #多帧图像文件处理

        if os.path.exists(Filepath + '/TIFFPaint'):  #创建文件夹用于保存每帧打点的图
            shutil.rmtree(Filepath + '/TIFFPaint')
            os.mkdir(Filepath + '/TIFFPaint')
        if not os.path.exists(Filepath + '/TIFFPaint'):
            os.mkdir(Filepath + '/TIFFPaint')

        for i, temp_img in enumerate(list(tif.iter_images())):  #对每帧图像打点，标记定位结果
            onePoint = locPoint[:, np.where(locPoint[12, :]==i+1)]
            onePoint = np.squeeze(onePoint)
            if onePoint.size==0: continue
            onePoint = np.array(onePoint, dtype=np.int0)
            onePoint = onePoint.reshape((13,-1))
            temp_img = np.array(temp_img, dtype=np.float32)
            for t in range(0, onePoint.shape[1]):
                cv2.circle(temp_img, (onePoint[0, t], onePoint[1, t]), radius=10, color=(0,0,255), thickness=1)
            cv2.imwrite(os.path.join(Filepath, 'TIFFPaint','%d.tif' %i), temp_img)
        tif.close()

        paintPath = os.path.join(Filepath, 'TIFFPaint')
        File_paint = glob(f'{paintPath}/*.tif', recursive=True)
        File_paint = sorted(File_paint)
        savetif = TIFF.open((portion[0] + '_Paint.tif'), mode='w')
        for tempPath in list(File_paint):   #将每帧打点结果保存为stack .tif文件
            tif = TIFF.open(tempPath, mode='r')
            img = tif.read_image()
            img = img.astype(np.uintc)
            img = Image.fromarray(img)
            savetif.write_image(img)
        savetif.close()


if __name__ == '__main__':

    Filename = r'E:\wgferh'
    name = '1.tif'
    locPaint2img(Filename, name, PSF_range=[1.1,10,1.1,10])