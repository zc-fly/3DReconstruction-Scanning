from storm_analysis.daostorm_3d.mufit_analysis import analyze
import numpy as np
import os
import scipy.io as sio
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
from glob import glob
import afterProcess.afterProcess as myFilter

def DAO_STORM(InputFilePath, OutputFilePath, ConfigPath,OutputFilePath_mat,FitMode=3):
    start = time.time()
    analyze(InputFilePath, OutputFilePath, ConfigPath)
    end = time.time()
    print("运行时间:%.2f秒" % (end - start))

    Dao3D_Result=np.vstack((1,1,1,1,1,1,1,1,1,1,1,1,1))
    Dao3D_Result=Dao3D_Result.astype(np.float64)

    import storm_analysis.sa_library.sa_h5py as saH5Py

    with saH5Py.SAH5Py(OutputFilePath) as h5:
        for fnum, locs in h5.localizationsIterator():
            x = locs['x']
            y = locs['y']
            xsigma = locs['xsigma']
            if(FitMode==3):
                z = locs['z']
                ysigma = locs['ysigma']
            height = locs['height']
            Sum = locs['sum']
            background = locs['background']
            category = locs['category']
            error = locs['error']
            iterations = locs['iterations']
            significance = locs['significance']
            frame_t = (fnum + 1)*np.ones([len(x)])
            if(FitMode==3):
                Dao3D_Result_t = np.vstack((x, y, z, xsigma, ysigma, height, Sum, background, category, error, iterations, significance, frame_t))
            else:
                Dao3D_Result_t = np.vstack((x, y, xsigma, height, Sum, background, category, error, iterations, significance, frame_t))
            if  Dao3D_Result_t.size == 0:
                continue
            else:
                Dao3D_Result = np.hstack((Dao3D_Result, Dao3D_Result_t))

    Dao3D_Result=Dao3D_Result[:,1:]
    mdict = {"Dao3D_Result": Dao3D_Result}
    sio.savemat(OutputFilePath_mat, mdict)

    return Dao3D_Result


def duplication_remove(arry,STEP=4000,PixelSize=165,r=1):
    # STEP = 4000         # nm
    # PixelSize = 165     # nm
    # r = 20              # pixel 假设是20个像素内

    arry = np.transpose(arry)   # 转置
    arry = np.insert(arry, 0, values=0, axis=1)
    arry[:,0] = np.arange(0,arry.shape[0])
    frameStart = arry[0, 13]
    frameEnd   = arry[-1,13]

    arryCopy = arry.copy()
    for fCur in range(int(frameStart+1),int(frameEnd+1)):
        fBefor = fCur-1
        fCurDataTemp   = arryCopy[arryCopy[:, 13] == fCur   , :]
        fBeforDataTemp = arryCopy[arryCopy[:, 13] == fBefor , :]

        if fCurDataTemp.size == 0  or fBeforDataTemp.size ==0 : continue
        id_start = fCurDataTemp[0, 0]
        id_end   = fCurDataTemp[-1,0]
        fCurData = arry[int(id_start):int(id_end+1),:]

        id_start = fBeforDataTemp[0, 0]
        id_end   = fBeforDataTemp[-1,0]
        fBeforData = arry[int(id_start):int(id_end+1),:]

        fCurDataCopy       = fCurData.copy()
        fCurDataCopy[:,2]  = fCurDataCopy[:,2] + STEP/PixelSize

        tree = spatial.cKDTree(fBeforData[:,1:3])
        distance,index = tree.query(fCurDataCopy[:,1:3],k=1)
        fCurData[distance < r,0] = fBeforData[index[distance < r],0]

    arry = arry.tolist()
    arry.sort(key=(lambda x:x[0]))
    arry = np.array(arry)

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

    return arry


def CurveFit(nameStr, arry, STEP, PixelSize, r):
    ''' 拼接去重方法
        输出结果 arry 的第一列是聚类后的标号，相同的数字表示是一个类别；

        后面可以根据 sigmax sigmay 等特征信息，从聚类后的结果中选择一个
        arry的第一列记录了数字信息，一次从小到大排列，相同的数字表示是不同帧，同一个分子
    '''
    arry = duplication_remove(arry, STEP, PixelSize, r)  #

    print('cluster的数量：%d', arry[-1, 0])

    # sigmaxy filter
    a = arry[:, 4] < sigmaxy_threadhold[1]
    b = arry[:, 4] > sigmaxy_threadhold[0]
    c = arry[:, 5] < sigmaxy_threadhold[1]
    d = arry[:, 5] > sigmaxy_threadhold[0]
    arry = arry[a * b * c * d, :]

    arry = arry[arry[:,12] > significance, :]

    _, indices, counts = np.unique(arry[:, 0], return_index=True, return_counts=True)
    indices = indices[counts > cluster_threadhold]
    counts = counts[counts > cluster_threadhold]

    arry_out = np.empty(shape=(0, 15))
    arry[:, 3] = arry[:, 4] ** 2 - arry[:, 5] ** 2
    indices = indices.tolist()
    for i, eachClass in enumerate(indices):
        classInfo = arry[eachClass:eachClass + counts[i], :]
        minIndex = np.argmin(abs(classInfo[:, 3]))
        if abs(classInfo[minIndex, 3]) < center_error:
            classInfo = np.insert(classInfo, -1, values=0, axis=1)
            curve_first = np.polyfit(classInfo[:, 3], classInfo[:, 14], 1)
            classInfo[:, 13] = (classInfo[:, 14] - curve_first[1]) * frame_step
            arry_out = np.vstack((arry_out, classInfo))

    if len(arry_out) == 0:
        print("No date valuble for fitting")

    def fitFun(x, p1, p2, p3, p4):
        return p1*x**3 + p2*x**2 + p3*x**1 + p4

    popt, pcov = curve_fit(fitFun, arry_out[:, 3], arry_out[:, 13])
    perr = np.sqrt(np.diag(pcov))
    sio.savemat(nameStr+'_curve.mat',mdict={'p1':popt[0], 'p2':popt[1], 'p3':popt[2], 'p4':popt[3]})
    print(popt)
    plt.figure()
    x = np.linspace(min(arry_out[:, 3]),max(arry_out[:, 3]) , 500)
    plt.plot(x, fitFun(x, *popt), 'r-')
    plt.scatter(arry_out[:, 3], arry_out[:, 13], marker='.', s=5, label='original datas')
    plt.legend()
    plt.show()

    return arry_out

def showTwoCurve(array1, array2, distance):
    curve1 = glob(f'camera1*_curve*.mat', recursive=True)
    curve2 = glob(f'camera2*_curve*.mat', recursive=True)
    calibs1 = sio.loadmat(curve1[0])
    calibs2 = sio.loadmat(curve2[0])

    def fitFun(x, p1, p2, p3, p4):
        return p1*x**3 + p2*x**2 + p3*x**1 + p4

    plt.figure()
    plt.scatter(array1[:, 3], array1[:, 13], marker='.', s=5, c='blue')
    plt.scatter(array2[:, 3], array2[:, 13]+distance, marker='.', s=5, c='darkred')

    x1 = np.linspace(min(array1[:, 3]),max(array1[:, 3]) , 500)
    x2 = np.linspace(min(array2[:, 3]), max(array2[:, 3]), 500)
    x1 = x1.reshape(-1, 500)
    x2 = x2.reshape(-1, 500)
    plt.plot(x1, fitFun(x1, calibs1['p1'], calibs1['p2'], calibs1['p3'], calibs1['p4']), 'r-')
    plt.plot(x2, fitFun(x2, calibs2['p1'], calibs2['p2'], calibs2['p3'], calibs2['p4']+distance), 'r-')
    plt.show()


if __name__ == '__main__':

    filename1 = r'H:\1-DataBackup\NPauto_Calibration_test\NP_Calibration\532\correct\T_ 2_25_calib_camera1_650_part_    0_correct'
    filename2 = r'H:\1-DataBackup\NPauto_Calibration_test\NP_Calibration\532\correct\T_ 2_25_calib_camera2_650_part_    0_correct'

    cluster_radius = 3
    cluster_threadhold = 50
    significance = 200
    sigmaxy_threadhold = [1.1, 8]
    frame_step = 100
    drift_correct = 0
    center_error = 50
    zdif_cameras = 3

    InputFile1 = filename1 + '.tif'
    OutputFilePath1 = filename1 + 'loc.hdf5'
    OutputFilePath_mat1 = filename1 + 'loc.mat'
    InputFile2 = filename2 + '.tif'
    OutputFilePath2 = filename2 + 'loc.hdf5'
    OutputFilePath_mat2 = filename2 + 'loc.mat'
    ConfigPath = r'Config_3d.xml'
    arry1 = DAO_STORM(InputFile1, OutputFilePath1, ConfigPath, OutputFilePath_mat1, FitMode=3)
    arry2 = DAO_STORM(InputFile2, OutputFilePath2, ConfigPath, OutputFilePath_mat2, FitMode=3)
    arry1 = CurveFit(filename1, arry1, STEP=drift_correct, PixelSize=165, r=cluster_radius)
    arry2 = CurveFit(filename2, arry2, STEP=drift_correct, PixelSize=165, r=cluster_radius)
