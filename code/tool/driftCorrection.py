from libtiff import TIFF
from PIL import Image
import numpy as np

step = -10/182#10nm/pixel_size
path = r'D:\A_CodeFile\python\MoTing\data\\'
cameraAll = ['ccut']

for cameraID, camera in enumerate(cameraAll):
    filepath = path + camera + '.tif'
    tif = TIFF.open(filepath, mode='r')
    img = tif.read_image(0)
    n = np.size(img, 0)
    m = np.size(img, 1)
    temp_top = n-2  # pixel
    temp_bottom = 1  # pixel
    AreaWidth =  temp_top - temp_bottom
    savetif = TIFF.open(path + '\\' + camera + '_correct.tif', mode='w')
    for img in list(tif.iter_images()):
        if temp_bottom<0:
            if temp_top<0:
                img = np.zeros([AreaWidth, m], dtype=np.uintc)
                img = Image.fromarray(img)
            else:
                tt = AreaWidth - int(abs(temp_bottom))
                img = img[0:tt][:]
                img = img.astype(np.uintc)
                temp = np.zeros([int(abs(temp_bottom)), m], dtype=np.uintc)
                img = np.vstack((temp, img))
                img = Image.fromarray(img)
        elif temp_top>n-1:
            if temp_bottom>n-1:
                img = np.zeros([AreaWidth, m], dtype=np.uintc)
                img = Image.fromarray(img)
            else:
                tt = AreaWidth - (int(temp_top)-n)
                img = img[int(temp_bottom):int(temp_bottom)+tt][:]
                img = img.astype(np.uintc)
                temp = np.zeros([int(temp_top)-n, m], dtype=np.uintc)
                img = np.vstack((img, temp))
                img = Image.fromarray(img)
        else:
            img = img[int(temp_bottom):int(temp_bottom) + AreaWidth][:]
            img = img.astype(np.uintc)
            img = Image.fromarray(img)

        # print(np.shape(img))
        savetif.write_image(img)
        temp_top = temp_top + step
        temp_bottom = temp_bottom + step
    print(camera + '-Complete!')
    savetif.close()

