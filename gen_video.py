import argparse
import numpy as np
import cv2
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--test_src', required=True)
    parser.add_argument('--test_result', required=True)
    parser.add_argument('--out_dir', required=True)
    return parser.parse_args()


def frame_generator(src_paths, result_paths):
    for src_path, result_path in zip(src_paths, result_paths):
        print src_path, result_path
        frame = np.zeros((128, 256, 3), dtype=np.uint8)
        src_im = cv2.imread(src_path)
        gt, prev, next = [src_im[:, 128 * i:128 * (i + 1)] for i in range(3)]
        result = cv2.imread(result_path)
        frame[:, :128] = prev
        frame[:, 128:] = prev
        yield frame
        frame[:, :128] = gt
        frame[:, 128:] = result
        yield frame


def filesuffix(path):
    return int(os.path.splitext(os.path.basename(path))[0])

args = parse_args()
src_paths = sorted(glob.glob(os.path.join(args.test_src, "*")), key=filesuffix)
result_paths = sorted(glob.glob(os.path.join(args.test_result, "*")))

g = frame_generator(src_paths, result_paths)
counter = 0
try:
    while True:
        cv2.imwrite(os.path.join(args.out_dir, "{:04d}.jpg".format(counter)), g.next())
        counter += 1

except StopIteration:
    pass
