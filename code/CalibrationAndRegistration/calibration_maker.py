# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import storm_analysis.sa_library.sa_h5py
from storm_analysis.sa_utilities import tracker
import storm_analysis.sa_library.sa_h5py as saH5Py


# class calibration_curve(MultiTrack):
#     """
#     input:多个tracker信息
#     output：利用storm_analyse包做好的一条标定曲线
#     """
#
#
#
#
# class registration_maker():
#     """
#     读取双相机定位结果，进行点云配准，获得两相机配准关系
#     """



def relationMaker(sa_hdf5_filepath):

    sa_hdf5_filename = glob.glob(os.path.join(sa_hdf5_filepath, '*.hdf5'))
    tracker.tracker(sa_hdf5_filename[0],radius=4)  #establish tracker
    a = saH5Py.loadTracks(sa_hdf5_filename[0])

    MultiTrack = {}
    with saH5Py.SAH5Py(sa_hdf5_filename[0]) as h5:
        for locs in h5.tracksIterator():
            temp = abs(locs['xsigma'] - locs['ysigma'])
            # z_off_frame = np.where(temp == min(temp))
            # locs['fname'] = locs['fname'] - z_off_frame
            # for field in locs:
            #     if field in MultiTrack:
            #         MultiTrack[field] = np.concatenate((MultiTrack[field], locs[field]))

    #转换tracker形式为标定曲线拟合的输入
    with saH5Py.SAH5Py(sa_hdf5_filename[0]) as h5:
        for locs in h5.tracksIterator():
            locs['fname'][max(locs['height']),:] - locs['fname'][locs['xsigma']-locs['']]




    # #做标定曲线
    # calibration_curve(MultiTrack)
    #
    #  #两相机配准
    # registration_maker()


if __name__ == '__main__':

    filename = r'D:\A_CodeFile\yiqiCode\meifishReconstruct3D\data\temp_test\locFile'
    relationMaker(filename)