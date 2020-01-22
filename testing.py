import os
import numpy as np
import xml.etree.ElementTree as Et

labels = {'aeroplane': 1, 'bicycle': 2,'bird': 3,'boat': 4,'bottle': 5,
           'bus': 6,'car': 7, 'cat': 8,'chair': 9,'cow': 10,
           'diningtable': 11, 'dog': 12,'horse': 13,'motorbike': 14,'person': 15,
           'pottedplant': 16, 'sheep': 17,'sofa': 18,'train': 19,'tvmonitor': 20}

base_dir = os.getcwd()
annot = base_dir + '\\train\\annot'

lbl = []

for e, i in enumerate(os.listdir(annot)):
    tree = Et.parse(annot + '\\' + i)
    root = tree.getroot()

    temp = []
    for member in root.findall('object'):
        name = member.find('name').text
        x1 = member.find('bndbox/xmin').text
        y1 = member.find('bndbox/ymin').text
        x2 = member.find('bndbox/xmax').text
        y2 = member.find('bndbox/ymax').text
        temp.append(labels[name])
        temp.append(x1)
        temp.append(x2)
        temp.append(y1)
        temp.append(y2)

    lbl.append(temp)

lbl = np.array(lbl)
print(lbl)