from storm_analysis.daostorm_3d.mufit_analysis import analyze
import numpy as np
import os
import scipy.io as sio
import multiprocessing
from libtiff import TIFFfile


def DAOstorm_outputtype2mat(InputFilePath, OutputFilePath, ConfigPath, OutputFilePath_mat, FitMode=3):
    '''
    Daostorm定位
    将Daostorm的定位结果.hdf5文件转为.mat文件
    '''
    analyze(InputFilePath, OutputFilePath, ConfigPath)

    Dao3D_Result = np.vstack((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    Dao3D_Result = Dao3D_Result.astype(np.float64)

    import storm_analysis.sa_library.sa_h5py as saH5Py

    with saH5Py.SAH5Py(OutputFilePath) as h5:
        for fnum, locs in h5.localizationsIterator():
            x = locs['x']
            y = locs['y']
            xsigma = locs['xsigma']
            if (FitMode == 3):
                z = locs['z']
                ysigma = locs['ysigma']
            height = locs['height']
            Sum = locs['sum']
            background = locs['background']
            category = locs['category']
            error = locs['error']
            iterations = locs['iterations']
            significance = locs['significance']
            frame_t = (fnum + 1) * np.ones([len(x)])
            if (FitMode == 3):
                Dao3D_Result_t = np.vstack((x, y, z, xsigma, ysigma, height, Sum, background, category, error,
                                            iterations, significance, frame_t))
            else:
                Dao3D_Result_t = np.vstack(
                    (x, y, xsigma, height, Sum, background, category, error, iterations, significance, frame_t))

            if Dao3D_Result_t.size == 0:
                continue
            else:
                Dao3D_Result = np.hstack((Dao3D_Result, Dao3D_Result_t))

    Dao3D_Result = Dao3D_Result[:, 1:]
    mdict = {"Dao3D_Result": Dao3D_Result}
    sio.savemat(OutputFilePath_mat, mdict)


def worker(Filelist, ProcessNum, ProcessID, ConfigPath, path, queue, FitMode=3):
    for stackID, stackFile in enumerate(Filelist):
        if stackID % ProcessNum == ProcessID:
            portion = os.path.splitext(stackFile)  # 将文件名拆成名字和后缀
            InputFilePath = os.path.join(path, stackFile)
            OutputFilePath = os.path.join(path, 'locFile', portion[0] + ".hdf5")  # 定位输出，存放在.hdf5文件中
            OutputFilePath_mat = os.path.join(path, 'locFile', portion[0] + ".mat")  # 定位输出，存放在.mat文件中
            DAOstorm_outputtype2mat(InputFilePath, OutputFilePath, ConfigPath, OutputFilePath_mat, FitMode=3)
            tif = TIFFfile(InputFilePath)  # 获取stack帧数
            stacksize = tif.get_depth()
            tif.close()
            queue.put((stackFile, stacksize))


def sequence_Loc_multiProcess(path):
    '''
    读取路径下的.tif文件
    调用analyse/DAOstorm_outputtype2mat对这些文件进行定位
    '''
    ConfigPath = "Config_3d.xml"
    MAXMUM_PROCESS = multiprocessing.cpu_count()

    if not os.path.exists(path + '/locFile'):
        os.mkdir(path + '/locFile')

    cameraAll = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.tif':
            cameraAll.append(file)

    if len(cameraAll) < MAXMUM_PROCESS:
        ProcessNum = len(cameraAll)
    else:
        ProcessNum = MAXMUM_PROCESS

    jobs = []
    queue = multiprocessing.Queue()
    for processID in range(0, ProcessNum):
        process= multiprocessing.Process(target=worker, args=(cameraAll,ProcessNum, processID, ConfigPath, path, queue))
        jobs.append(process)
        process.start()
    for process in jobs:
        process.join()

    frameList = []
    for l in range(0,queue.qsize()):
        frameList.append(queue.get())

    with open(path + '/locFile/frameInfo.txt', 'w') as f:
        for i in list(frameList):
            f.write(i[0]+' '+str(i[1])+'\n')



def sequence_Loc_oneFile(Filename):
    '''
    读取指定.tif文件
    调用analyse/DAOstorm_outputtype2mat对这其进行定位
    '''
    ConfigPath = "Config_3d.xml"
    portion = os.path.splitext(Filename)
    if os.path.exists(Filename):
        OutputFilePath = os.path.join(portion[0] + ".hdf5")
        OutputFilePath_mat = os.path.join(portion[0] + ".mat")
        DAOstorm_outputtype2mat(Filename, OutputFilePath, ConfigPath, OutputFilePath_mat, FitMode=3)


if __name__ == '__main__':

    path = r'C:\Users\Mr Xin\Desktop\resolutuionTest'
    sequence_Loc_multiProcess(path)