"""
This script main use to remove duplicate points from error loclization and sequence frames

duplication_remove()
去除双相机连续帧重复点

radius_filter()
滤除半径范围内的点。主要用于剔除错误定位点。

"""
import numpy as np
from scipy import spatial
import open3d as o3d


def calssFilter_condition(array_in):
    """
    :argument:对聚类完的点进行筛选，保留单个点，一般可以根据峰值信号筛选(内部函数)
    :param array:
    :return:
    """
    array_out = np.empty(shape=(13, 0))
    array_in = array_in.transpose()
    Frameslist = (array_in[0,:]).tolist()
    Frameslist = list(set(Frameslist))
    for i in Frameslist:
        temp = array_in[:, array_in[0,:]==i]
        peak_list = (temp[6, :]).tolist()
        index = peak_list.index(max(peak_list))
        temp = temp[1:14, index] #x y z frame
        temp = temp.reshape((13,-1))
        array_out = np.hstack((array_out, temp))
    return array_out


def duplication_remove(arry, displaceInfo, PixelSize=165, r=10):
    """
    :argument: 去除双相机连续帧重复点
    :param arry: 13xN numpy array /(x, y, z, xsigma, ysigma, height, Sum, background, category, error,iterations, significance, frame_t)
    :param displaceInfo: 平移台导出的位移表
    :param PixelSize：相机pixelsize
    :param r: 去重搜索范围
    :return: 4xN numpy array /(x, y, z, frame)
    """
    arry = np.transpose(arry)   # 转置
    # 【插入clusterID】
    arry = np.insert(arry, 0, values=0, axis=1)
    arry[:,0] = np.arange(0,arry.shape[0])

    ## 【获取j及j+1张定位点】
    frameStart = 1
    frameEnd   = arry[-1,13]

    arryCopy = arry.copy()
    for fCur in range(frameStart+1,int(frameEnd+1)):     # 会一直读取到 frameEnd -1
        fBefor = fCur-1

        # 读取当前帧、下一帧的数据；此操作后，clusterID变了。加入了arryCopy = arry.copy()调整bug
        fCurDataTemp   = arryCopy[arryCopy[:, 13] == fCur   , :]    # 复制操作
        fBeforDataTemp = arryCopy[arryCopy[:, 13] == fBefor , :]
        if (fCurDataTemp.shape[0] == 0) or (fBeforDataTemp.shape[0] == 0): continue

        id_start = fCurDataTemp[0, 0]                               # 期望fCurData 是 arry 部分的浅拷贝
        id_end   = fCurDataTemp[-1,0]
        fCurData = arry[int(id_start):int(id_end+1),:]

        id_start = fBeforDataTemp[0, 0]                     # 期望fBeforData 是 arry 部分的浅拷贝
        id_end   = fBeforDataTemp[-1,0]
        fBeforData = arry[int(id_start):int(id_end+1),:]

        # 先变换到同一个坐标系
        fCurDataCopy = fCurData.copy()
        # fCurDataCopy[:,2]  = fCurDataCopy[:,2] + 1000000*(float(displaceInfo[fCur, 0])-float(displaceInfo[fBefor, 0])) / PixelSize

        tree = spatial.cKDTree(fBeforData[:,1:3])
        # 查找fCurDataCopy中每个点在fBeforData中的最近邻
        distance,index = tree.query(fCurDataCopy[:,1:3],k=1)

        # 将fBeforData中查找到距离小于r的最近邻点，的clusterID 赋值给 fCurData
        fCurData[distance < r,0] = fBeforData[index[distance < r],0]

    # 先从前往后，按照从小到大的顺序行排序
    arry = arry.tolist()
    arry.sort(key=(lambda x:x[0]))
    arry = np.array(arry)

    # 从0开始，依次标记clusterID
    clusterID = 0
    befValueTemp = arry[0, 0]
    for curIIndex in range(1,arry.shape[0]):
        befIIndex = curIIndex - 1

        curValue  = arry[curIIndex, 0]
        befValue  = arry[befIIndex, 0]

        if curValue != befValueTemp:
            clusterID += 1
            befValueTemp = arry[curIIndex, 0]
            arry[curIIndex, 0] = clusterID
        else:
            arry[curIIndex, 0] = clusterID

    # arry_draw = np.empty(shape=(3, 0))
    array_draw = calssFilter_condition(array_in =arry)

    return array_draw


def radius_filter(sourcePoint, radius=5):
    """
    :argument: 滤除半径范围内的点。主要用于剔除错误定位点。
    :param pointCloud: 13xN numpy array /(x, y, z, xsigma, ysigma, height, Sum, background, category, error,iterations, significance, frame_t)
    :param radius: search radius(um)
    :param min_Points: Minimum number of points to form a cluster
    :return: 13xN numpy array /(x, y, z, xsigma, ysigma, height, Sum, background, category, error,iterations, significance, frame_t)
    """
    min_Points = 2
    pointCloud = sourcePoint.copy()
    pointCloud3D = pointCloud[0:3, :]
    pointCloud3D[2, :] = pointCloud3D[2, :] * 0
    pointCloud3D = pointCloud3D.transpose().tolist()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCloud3D)
    labels = np.array(pcd.cluster_dbscan(eps=radius, min_points=min_Points))  # 获取聚类标签
    for i in range(0,max(labels)):
        index = np.argwhere(labels==i)
        findmax = pointCloud[5,index].tolist()
        temp = findmax.index(max(findmax))
        labels[index[temp]]=-1

    return labels


# if __name__ == '__main__':

    # ## 【输入】
    # InputFilePath   = r'T_ 4_30_MultiFov_camera1_1250_part_0-2.tif'
    # OutputFilePath  = r'result.hdf5'
    # ConfigPath      = r'Config_3d.xml'
    # OutputFilePath_mat = r'result.mat'
    # arry = DAO_STORM(InputFilePath, OutputFilePath, ConfigPath,OutputFilePath_mat,FitMode=3)
    #
    #
    # # STEP 是步长，PixelSize是原始图尺寸，r是查找半径
    # STEP = 4000
    # PixelSize = 165
    # r = 20
    #
    # ''' 拼接去重方法
    # 输出：
    # 1） arry矩阵
    # clusterID, x, y, z, xsigma, ysigma, height, Sum, background, category, error, iterations, significance, frame_t
    # arry矩阵的第一列（clusterID）是聚类后的标号，相同的数字表示是一个类别；
    # arry矩阵的第一列（clusterID）是从小到大依次递增
    # '''
    #
    # arry = duplication_remove(arry,STEP,PixelSize,r)
    #
    # print('cluster的数量：%d',arry[-1,0])
    #
    #
    #
    # ''':验证正确性
    # 选了前四张图，分别用数字标识每个分子，相同的数字表示是同一个信号。
    # 因为相同信号出现在不同的帧，目标是就是将这些不同帧中的同一个信号给标识出来。本方法使用
    # 同一个数字标识，表示是同一个分子。
    #
    # 标识信息来自于arry的第一列
    # arry = duplication_remove(arry,STEP=4000,PixelSize=165,r=20)
    #
    # arry的第一列记录了数字信息，一次从小到大排列，相同的数字表示是不同帧，同一个分子
    #
    # '''
    # img1 = cv2.imread(r'.\image\0000.tif',cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imread(r'.\image\0001.tif',cv2.IMREAD_UNCHANGED)
    # img3 = cv2.imread(r'.\image\0002.tif',cv2.IMREAD_UNCHANGED)
    # img4 = cv2.imread(r'.\image\0003.tif',cv2.IMREAD_UNCHANGED)
    #
    # plt.figure(figsize=(10, 8))
    # plt.subplot(4,1,1)
    # plt.imshow(img1, cmap='gray',vmin=0,vmax=645)
    #
    # frame = 1
    # dataAll = arry[arry[:,13] == frame,:]
    # for data in dataAll:
    #     plt.text(data[1],data[2],str(data[0]),ha = 'center',color = "r",size = 8)
    #
    # plt.subplot(4,1,2)
    # plt.imshow(img2, cmap='gray',vmin=0,vmax=645)
    # frame = 2
    # dataAll = arry[arry[:,13] == frame,:]
    # for data in dataAll:
    #     plt.text(data[1],data[2],str(data[0]),ha = 'center',color = "r",size = 8)
    #
    # plt.subplot(4,1,3)
    # plt.imshow(img3, cmap='gray',vmin=0,vmax=645)
    #
    # frame = 3
    # dataAll = arry[arry[:,13] == frame,:]
    # for data in dataAll:
    #     plt.text(data[1],data[2],str(data[0]),ha = 'center',color = "r",size = 8)
    #
    # plt.subplot(4,1,4)
    # plt.imshow(img4, cmap='gray',vmin=0,vmax=645)
    # frame = 4
    # dataAll = arry[arry[:,13] == frame,:]
    # for data in dataAll:
    #     plt.text(data[1],data[2],str(data[0]),ha = 'center',color = "r",size = 8)
    #
    # plt.show()


