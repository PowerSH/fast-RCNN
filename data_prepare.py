import os
import shutil

base_dir = os.getcwd()
base_img_dir = os.path.join(base_dir, "JPEGImages_Sample")
base_annot_dir = os.path.join(base_dir, "Annotations_Sample")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

numbering = []
for i, e in enumerate(os.listdir(base_annot_dir)):
    e = e.split('.xml')
    numbering.append(e)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(500)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(train_dir, im)
    shutil.copyfile(src, dst)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(500, 750)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(valid_dir, im)
    shutil.copyfile(src, dst)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(750, 1000)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(test_dir, im)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(500)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(train_dir, at)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(500, 750)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(valid_dir, at)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(750, 1000)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(test_dir, at)
    shutil.copyfile(src, dst)