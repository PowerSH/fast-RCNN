import os
import cv2
import xml.etree.ElementTree as Et
import data_prepare as dp

labels = {'aeroplane': 1, 'bicycle': 2,'bird': 3,'boat': 4,'bottle': 5,
           'bus': 6,'car': 7, 'cat': 8,'chair': 9,'cow': 10,
           'diningtable': 11, 'dog': 12,'horse': 13,'motorbike': 14,'person': 15,
           'pottedplant': 16, 'sheep': 17,'sofa': 18,'train': 19,'tvmonitor': 20}

base_dir = os.getcwd()
annot = base_dir + '\\train\\annot'

def labeling(dir):
    lbl = []
    for e, i in enumerate(os.listdir(annot)):
        tree = Et.parse(annot + '\\' + i)
        root = tree.getroot()

        temp = []
        for member in root.findall('object'):
            name = member.find('name').text
            temp.append(labels[name])

        lbl.append(temp)
    return lbl

def dataprepare(dir):
    X = []
    for e, i in enumerate(os.listdir(dp.train_image_dir)):
        img = cv2.imread(dp.train_image_dir + '\\' + i)
        X.append(img)
    return X