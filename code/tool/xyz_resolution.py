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


def psf_stastitic(path, pixelsize, see_frame=2, PSF_range=[1.01,10,1.01,10]):
    # 该函数的目的是统计分析成像的xy轴分辨率。由于成像的分辨率取决于点 PSF的大小，我们对不同深度点的PSF进行了统计。
    # 这里我们对重建好的三维点进行分析，统计了全部三维点在不同深度下的PSF值(以max{sigmax,sigmay}作为某点的PSF)

    #resolution threshold
    xy_resolution_1 = 1 #um 半峰宽我们认为等于xy_resolution时为能够分辨的极限
    xy_resolution_2 = 2  # um 半峰宽我们认为等于xy_resolution时为能够分辨的极限

    locTool.sequence_Loc_multiProcess(path)
    loc_point = load_locfile(path)
    loc_point = loc_point[:, (loc_point[3] > PSF_range[0]) & (loc_point[3] < PSF_range[1])]  # sigmax filter
    loc_point = loc_point[:, (loc_point[4] > PSF_range[2]) & (loc_point[4] < PSF_range[3])]  # sigmay filter

    loc_point = loc_point[:, loc_point[12, :]==see_frame]    #选择某一帧统计psf信息
    maxsigma = np.amax(loc_point[2:4, :], axis=0)
    maxsigma = 2.35*maxsigma * pixelsize / 1000

    a1 = np.size(maxsigma[maxsigma<xy_resolution_1])
    a2 = np.size(maxsigma[maxsigma < xy_resolution_2])
    percent1 = a1 / np.size(maxsigma)
    percent2 = a2 / np.size(maxsigma)

    #直方图
    fig = plt.figure()
    ax = fig.add_subplot(211)
    numBins = 20
    ax.hist(maxsigma, numBins, color='blue', alpha=0.8, rwidth=0.9)
    plt.grid(True)
    plt.xlabel("PSF-HalfWidth/um")
    plt.ylabel("counts")
    plt.axvline(x=xy_resolution_1, c="y", ls="--", lw=2, label='percent:{:.2f}%'.format(100*percent1))
    plt.axvline(x=xy_resolution_2, c="r", ls="--", lw=2, label='percent:{:.2f}%'.format(100*percent2))
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
    plt.scatter(onePoint[0,:], onePoint[1,:], s=10, c='none', marker='o', edgecolors='r')
    plt.imshow(img, cmap='gist_gray')
    plt.show()


def z_analyse_compareshapeVary(path_z_analyse, Thresh_hold=200, startFrame=20, totalFrames = 4, stepFrame=3):
    """
    :argument: 分析z轴分辨率
    :param path_z_analyse: 图像路径
    :param Thresh_hold: 图像阈值化阈值设置
    :param startFrame: 起始帧数
    :param totalFrames: 希望展示帧数
    :param stepFrame: 帧间隔
    """
    camera_zrange = [-4, 4]
    path = path_z_analyse
    locTool.sequence_Loc_multiProcess(path)
    loc_point = load_locfile(path)  #定位结果
    File_img = glob(f'{path}/*.tif', recursive=True)    #原始图像
    curve = glob(f'{path}/*curve.mat', recursive=True)  #加载标定曲线
    if len(curve) == 0:
        print("check your curve.mat File!")
        exit(0)
    calibs = sio.loadmat(curve[0])

    for j in range(0, loc_point.shape[1]):
        sigmaxy = loc_point[3][j] ** 2 - loc_point[4][j] ** 2 ##根据标定曲线重建z轴信息
        loc_point[2][j] = calibs['p1'] * sigmaxy ** 4 + calibs['p2'] * sigmaxy ** 3 + calibs['p3'] * sigmaxy ** 2 + calibs['p4'] * sigmaxy
        if loc_point[2][j] < camera_zrange[0] or loc_point[2][j] > camera_zrange[1]: loc_point[2][j] = 404   #剔除重建错误的点
    loc_point = loc_point[:, loc_point[2,:]!=404]

    #读取图像
    tif = TIFF.open(File_img[0], mode='r')
    Imagelist_Threath = []
    Imagelist_raw = []
    for img in list(tif.iter_images()):
        Imagelist_raw.append(img)
        img = np.where(img > Thresh_hold, 0, 100)
        Imagelist_Threath.append(img)

    #show vary
    count = 0
    for i in range(startFrame, startFrame+stepFrame*totalFrames, stepFrame):
        count = count + 1
        if count>totalFrames: break
        plt.subplot(2, totalFrames, count)
        plt.imshow(Imagelist_raw[i])
        plt.subplot(2,totalFrames, totalFrames +count)
        plt.imshow(Imagelist_Threath[i])
        temp = loc_point[:, loc_point[12, :] == i]
        if temp.size==0:
            print("some frames lack! change start frame!")
            exit(0)
        infoarray = loc_point[[2,3,4],(loc_point[12,:]==i)] #z,sigmax,sigmay
        if infoarray.size !=3:
            print("change start frame!")
            exit(0)
        plt.xlabel("Z-Depth: %.2fμm" %infoarray[0])
        plt.title('Half-With(%.2f, %.2f)' %(float(2.35*infoarray[1]),float(infoarray[2]*2.35)))
    plt.show()



if __name__ == '__main__':

    path_xy_analyse = r'I:\temp'
    path_z_analyse = r'D:\A_CodeFile\yiqiCode\meifishReconstruct3D\data\xyz_target\z_analyse'

    psf_stastitic(path_xy_analyse, pixelsize=162.5, see_frame=1, PSF_range=[1.1,5,1.1,5]) #xy分辨率分析\计算PSF半峰宽
    # z_analyse_compareshapeVary(path_z_analyse, Thresh_hold=3300, startFrame=140, totalFrames=5, stepFrame=6)    #比较z方向的形变
