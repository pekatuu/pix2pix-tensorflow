import argparse
import glob
import os
import numpy as np
import scipy.misc


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mot_root', required=True)
    parser.add_argument('--img_size', default=128)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--step', default=5)
    return parser.parse_args()


def find_images(base_dir):
    return sorted(glob.glob(os.path.join(base_dir, "*.jpg")))


def gen_image_path(base_dir, step):
    image_paths = find_images(base_dir)
    for path in image_paths[::step]:
        yield path


def gen_out_name(output_dir, start=1):
    i = start
    while True:
        yield os.path.join(output_dir, "{}.jpg".format(i))
        i += 1


def convert_dir(img_dir, out_name_generator, img_size, step):
    g = gen_image_path(img_dir, step)
    try:
        while True:
            out_image = np.empty((img_size, img_size * 3, 3), dtype=np.uint8)
            a1 = g.next()
            b = g.next()
            a2 = g.next()
            for i, path in enumerate([b, a1, a2]):
                im = scipy.misc.imresize(scipy.misc.imread(path), (img_size, img_size))
                out_image[:, img_size * i:img_size * (i + 1), :] = im
            out_name = out_name_generator.next()
            scipy.misc.imsave(out_name, out_image)
            print("join {} {} {} into {}".format(a1, a2, b, out_name))
    except StopIteration:
        pass


def mkdirp(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def convert_mot(args):
    mot_root = args.mot_root

    test_dirs = glob.glob(os.path.join(mot_root, 'test', '*', 'img1'))
    convert_dirs(args, "test", test_dirs)

    trainval_dirs = glob.glob(os.path.join(mot_root, 'train', '*', 'img1'))
    dir_num = int(len(trainval_dirs) * 0.7)
    convert_dirs(args, "train", trainval_dirs[:dir_num])
    convert_dirs(args, "val", trainval_dirs[dir_num:])


def convert_dirs(args, dir_name, dirs):
    out_dir_train = os.path.join(args.output_dir, dir_name)
    mkdirp(out_dir_train)
    train_name_generator = gen_out_name(out_dir_train)
    for train_dir in dirs:
        convert_dir(train_dir, train_name_generator, args.img_size, args.step)


if __name__ == '__main__':
    convert_mot(parse_args())
