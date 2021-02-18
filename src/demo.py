from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def combine_images(left, right):
  rows_rgb, cols_rgb, channels = left.shape
  rows_gray, cols_gray, _ = right.shape
  rows_comb = max(rows_rgb, rows_gray)
  cols_comb = cols_rgb + cols_gray
  comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
  comb[:rows_rgb, :cols_rgb] = left
  comb[:rows_gray, cols_rgb:] = right

  return comb


def undistort_image(img):
    """
    A custom function to undistort image
    just for Negeley-Black Video.
    """
    h, w = img.shape[:2]
    mtx = np.array([
        [3389.14855, 0, 982.985434],
        [0, 3784.14471, 556.363307],
        [0, 0, 1]]
    )
    dist = np.array([-1.83418584,  12.2930625, -0.00434882103,  0.0226389517, -85.1805652])

    # undistort
    img = cv2.undistort(img, mtx, dist)
    return img


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  if not os.path.exists('./results'):
    os.makedirs('./results')

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    out_name = (opt.demo.split('/')[-1]).split('.')[0]
    detector.pause = False
    cnt = 0
    while True:
        _, img = cam.read()
        if img is None:
            try:
                out.release()
            except:
                print('File not found!!!')
            return
        cnt += 1
        # cv2.imshow('input', img)
        img = undistort_image(img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print('Frame ' + str(cnt) + ' |' + time_str)
        img = combine_images(ret['add_pred'], ret['bird_pred'])
        if cnt == 1:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('./results/{}_output.avi'.format(out_name), 
                    fourcc, 10, (img.shape[1], img.shape[0]))
        out.write(img)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      img_name = (image_name.split('/')[-1]).split('.')[0]
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      cv2.imwrite('./results' + '/{}_add_pred.png'.format(img_name), ret['add_pred'])
      cv2.imwrite('./results' + '/{}_bird_pred.png'.format(img_name), ret['bird_pred'])

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
