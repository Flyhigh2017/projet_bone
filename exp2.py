import numpy as np
import cv2
from os import listdir
import tensorflow as tf
import codecs, json
def contrast(img, clipLimit = 3.0):
    def max_to_js(mat, path):
        temp = mat.tolist()
        return json.dump(temp, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=(3,3))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    
    start_matrix = [['xmin', 'ymin', 'xmax', 'ymax', 'image_name', 'class']]
    fmt='%s %s %s %s %s %s'
    column_vec = np.zeros(shape=(1,6)).astype(np.str)
    column_vec[0,0] = '3'
    column_vec[0,1] = '3'
    column_vec[0,2] = '30'
    column_vec[0,3] = '30'
    column_vec[0,4] = 'try.jpg'
    column_vec[0,5] = 'bone'
    start_matrix = np.row_stack((start_matrix,column_vec))
    np.savetxt('label.txt', start_matrix, fmt = fmt)#can specify a data path
    b = start_matrix.tolist()
    file_path1 = './image.json'
    file_path2 = './label.json'
    print max_to_js(img2,file_path1)
    print max_to_js(start_matrix,file_path2)

'''
data_path = '/Users/anekisei/Documents/experiment/try.jpg'
im = cv2.imread(data_path)
contrast(im)

file_path = '/Users/anekisei/Documents/experiment/path.json'
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)
print a_new
'''
