import os
from libtiff import TIFFfile


path = r'\\192.168.1.102\cl2\20211227\A3470'

cameraAll = []
for file in os.listdir(path):
    if os.path.splitext(file)[1] == '.tif':
        cameraAll.append(file)

frameList = []
count = 0
for stackID, stackFile in enumerate(cameraAll):
    InputFilePath = os.path.join(path, stackFile)
    tif = TIFFfile(InputFilePath)  # 获取stack帧数
    stacksize = tif.get_depth()
    tif.close()
    frameList.append((stackFile, stacksize))
    print(count)
    count = count +1

with open(path + '/locFile/frameInfo.txt', 'w') as f:
    for i in list(frameList):
        f.write(i[0]+' '+str(i[1])+'\n')