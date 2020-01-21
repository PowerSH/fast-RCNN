import os
import shutil

base_dir = os.getcwd()
base_img_dir = os.path.join(base_dir, "JPEGImages_Sample")
base_annot_dir = os.path.join(base_dir, "Annotations_Sample")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

train_image_dir = train_dir + '\\image'
test_image_dir = test_dir + '\\image'
valid_image_dir = valid_dir + '\\image'

train_annot_dir = train_dir + '\\annot'
test_annot_dir = test_dir + '\\annot'
valid_annot_dir = valid_dir + '\\annot'

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

if not os.path.exists(train_dir + '/image'):
    os.mkdir(train_dir + '/image')

if not os.path.exists(test_dir + '/image'):
    os.mkdir(test_dir + '/image')

if not os.path.exists(valid_dir + '/image'):
    os.mkdir(valid_dir + '/image')

if not os.path.exists(train_dir + '/annot'):
    os.mkdir(train_dir + '/annot')

if not os.path.exists(test_dir + '/annot'):
    os.mkdir(test_dir + '/annot')

if not os.path.exists(valid_dir + '/annot'):
    os.mkdir(valid_dir + '/annot')

numbering = []
for i, e in enumerate(os.listdir(base_annot_dir)):
    e = e.split('.xml')
    numbering.append(e)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(500)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(train_dir + '/image', im)
    shutil.copyfile(src, dst)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(500, 750)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(valid_dir + '/image', im)
    shutil.copyfile(src, dst)

img_name = ['{}.jpg'.format(numbering[i][0]) for i in range(750, 1000)]
for im in img_name:
    src = os.path.join(base_img_dir, im)
    dst = os.path.join(test_dir + '/image', im)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(500)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(train_dir + '/annot', at)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(500, 750)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(valid_dir + '/annot', at)
    shutil.copyfile(src, dst)

annot_name = ['{}.xml'.format(numbering[i][0]) for i in range(750, 1000)]
for at in annot_name:
    src = os.path.join(base_annot_dir, at)
    dst = os.path.join(test_dir + '/annot', at)
    shutil.copyfile(src, dst)