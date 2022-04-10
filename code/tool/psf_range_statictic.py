from libtiff import TIFF
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import utils.loc_utils as locTool
from glob import glob
import build3D.Reconstruction3D as buildTool


def file_filter(f):
    if f[-4:] in ['.mat']:
        return True
    else:
        return False


def load_locfile(path):

    path_loc = path + '\\locFile'  # 定位表路径,分别筛选出两个相机定位表
    locFile = glob(f'{path_loc}/*.mat', recursive=True)
    locPoint = sio.loadmat(locFile[0])
    locPoint = locPoint['Dao3D_Result']
    return locPoint


def psf_stastitic(path,open_psf_filter,PSF_range, see_frame=0):
    # 该函数的目的是统计分析成像的xy轴分辨率。由于成像的分辨率取决于点 PSF的大小，我们对不同深度点的PSF进行了统计。
    # 这里我们对重建好的三维点进行分析，统计了全部三维点在不同深度下的PSF值(以max{sigmax,sigmay}作为某点的PSF)

    locTool.sequence_Loc_multiProcess(path)
    loc_point = load_locfile(path)
    if open_psf_filter==1:
        loc_point = loc_point[:,(loc_point[3,:] > PSF_range[0]) & (loc_point[3,:] < PSF_range[1])]  # sigmax filter
        loc_point = loc_point[:,(loc_point[4,:] > PSF_range[2]) & (loc_point[4,:] < PSF_range[3])]  # sigmay filter
    loc_point = loc_point[:, loc_point[12, :]==see_frame]    #选择某一帧统计psf信息
    maxsigma = np.amax(loc_point[2:4, :], axis=0)

    #直方图
    fig = plt.figure()
    ax = fig.add_subplot(211)
    numBins = 20
    ax.hist(maxsigma, numBins, color='blue', alpha=0.8, rwidth=0.9)
    plt.grid(True)
    plt.xlabel("PSF-HalfWidth/um")
    plt.ylabel("counts")
    plt.legend()

    #打点图
    ax = fig.add_subplot(212)
    Filename = glob(f'{path}/*.tif', recursive=True)
    tif = TIFF.open(Filename[0], mode='r')
    count = 1
    for img in list(tif.iter_images()):
        if count ==see_frame: break
        count+=1

    onePoint = np.array(loc_point[[0,1],:])  # 转换为opencv可读数据类型
    img = np.array(img, dtype=np.float32)  # 转换为opencv可读数据类型
    plt.scatter(onePoint[0,:], onePoint[1,:], s=35, c='none', marker='o', edgecolors='r')
    plt.imshow(img, cmap='gist_gray')
    plt.show()



if __name__ == '__main__':

    path_xy_analyse = r'\\192.168.1.102\cl\20220224\532\scan\1\132'

    psf_stastitic(path_xy_analyse,open_psf_filter=1, PSF_range=[1.1,3,1.1,5], see_frame=1) #xy分辨率分析\计算PSF半峰宽
    # 如果設置open_psf_filter=1，可以觀看daostorm定位后，且經過你設置的psf濾波后的打點統計圖。設置為0，則只顯示daostorm定位后的打點結果