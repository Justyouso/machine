# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-5-10 下午3:22


import os,shutil

# 源数据
# original_dataset_dir = '/home/justyouso/space/data/machine/dogs-vs-cats/train/'
#
# train_dogs_dir = "/home/justyouso/space/data/machine/dogs-cats/train/dogs"
# train_cats_dir = "/home/justyouso/space/data/machine/dogs-cats/train/cats"
# validation_dogs_dir = "/home/justyouso/space/data/machine/dogs-cats/validation/dogs"
# validation_cats_dir = "/home/justyouso/space/data/machine/dogs-cats/validation/cats"
# test_dogs_dir = "/home/justyouso/space/data/machine/dogs-cats/test/dogs"
# test_cats_dir = "/home/justyouso/space/data/machine/dogs-cats/test/cats"

base_dir = '/workspace/data/machine'
original_dataset_dir = base_dir + '/dogs-vs-cats/train/'

train_dogs_dir = base_dir + "/dogs-cats/train/dogs"
train_cats_dir = base_dir + "/dogs-cats/train/cats"
validation_dogs_dir = base_dir + "/dogs-cats/validation/dogs"
validation_cats_dir = base_dir + "/dogs-cats/validation/cats"
test_dogs_dir = base_dir + "/dogs-cats/test/dogs"
test_cats_dir = base_dir + "/dogs-cats/test/cats"

# dogs训练数据(1000)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# dogs验证数据(500)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# dogs测试数据(500)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# cats训练数据(1000)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# cats验证数据(500)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# cats测试数据(500)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
