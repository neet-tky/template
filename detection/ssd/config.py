# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

voc = {
    'class_num': 21,
    'lr_steps': [80000, 100000, 120000],
    'max_iter': 120000,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratio': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [.1, .2],
    'name': 'VOC',
}