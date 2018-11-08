import os 
import glob
from PIL import Image, ImageFont, ImageDraw
import numpy as np

dir = '/home/admin1/sz/onsetdetection/pic/object_detection/*/*cqt.jpg'
filename = 'train.txt'


def get_info():
    jpgs = [jpg for jpg in glob.glob(dir)]
    jpg_info = []
    for jpg in jpgs:
        jpg_name = os.path.split(jpg)[1]
        jpg_txt = os.path.splitext(jpg_name)[0][:-4]
        jpg_onset = jpg_txt.split('--')[1][1:-1].split(', ')
        if jpg_onset[0]=='': 
            jpg_info.append((jpg, []))
            continue
        jpg_onset = [int(416*float(i)) for i in jpg_onset]
        jpg_box = []
        for i in jpg_onset:
            if i<8: 
                jpg_box.append((0, 0, i+8, 415))    # TODO： 这个是从0～415, 还是应该1～416
                continue
            if i>(415-8):
                jpg_box.append((i-8, 0, 415, 415))
                continue
            jpg_box.append((i-8, 0, i+8, 415))
        jpg_info.append((jpg, jpg_box))
    return jpg_info

def show():
    jpgs_info = get_info()
    for k in np.arange(0, 6000, 123):
        # if not k==5342: continue
        print(len(jpgs_info), jpgs_info[k])

        image = Image.open(jpgs_info[k][0])
        draw = ImageDraw.Draw(image)
        for i in range(len(jpgs_info[k][1])):
            left, bottom, right, top = jpgs_info[k][1][i]
            for i in range(2):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline='red')
        image.show(title='sdlfj')

        image_cqt = Image.open(jpgs_info[k][0][:-7]+'mid.jpg')
        draw = ImageDraw.Draw(image_cqt)
        for i in range(len(jpgs_info[k][1])):
            left, bottom, right, top = jpgs_info[k][1][i]
            for i in range(2):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline='red')
        image_cqt.show(title='sdlfj')
        input('input any:')


f = open(filename, 'w')
jpgs_info = get_info()
for jpg_info in jpgs_info:
    jpg_dir = jpg_info[0]
    bboxs = jpg_info[1]
    jpg_box_str = ''
    for bbox in bboxs:
        for i in bbox: jpg_box_str += (str(i) + ',')
        jpg_box_str += '0||'
    line = '\n'+ jpg_dir+ '||' + jpg_box_str
    line = line[:-2]
    f.writelines(line)
f.close()
