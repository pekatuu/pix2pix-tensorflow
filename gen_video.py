import argparse
import numpy as np
import cv2
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--test_src', required=True)
    parser.add_argument('--test_result', required=True)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_video', required=True)
    parser.add_argument('--data_per_image', required=True, type=int, default=3)
    return parser.parse_args()


def frame_generator(src_paths, result_paths, data_per_image):
    for src_path, result_path in zip(src_paths, result_paths):
        print src_path, result_path
        frame = np.zeros((128, 256, 3), dtype=np.uint8)
        src_im = cv2.imread(src_path)
        src_imgs = [src_im[:, 128 * i:128 * (i + 1)] for i in range(data_per_image)]
        gts = src_imgs[:-2]
        prev, next = src_imgs[-2:]
        result_ = cv2.imread(result_path)
        interp_result = [result_[:, 128 * i:128 * (i + 1)] for i in range(data_per_image - 2)]
        frame[:, :128] = prev
        frame[:, 128:] = prev
        yield frame
        for i in range(data_per_image - 2):
            frame[:, :128] = gts[i]
            frame[:, 128:] = interp_result[i]
            yield frame


def filesuffix(path):
    return int(os.path.splitext(os.path.basename(path))[0])

args = parse_args()
src_paths = sorted(glob.glob(os.path.join(args.test_src, "*")), key=filesuffix)
result_paths = sorted(glob.glob(os.path.join(args.test_result, "*")))
fourcc = cv2.cv.FOURCC(*'XVID')
result_video = cv2.VideoWriter(args.out_video, fourcc, 10, (256, 128))

g = frame_generator(src_paths, result_paths, args.data_per_image)
counter = 0
try:
    while True:
        im = g.next()
        if args.out_dir:
            cv2.imwrite(os.path.join(args.out_dir, "{:04d}.jpg".format(counter)), im)
        result_video.write(im)
        counter += 1

except StopIteration:
    pass

result_video.release()