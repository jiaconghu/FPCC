import os
import shutil


def walk_file(path):
    count = 0
    for root, dirs, files in os.walk(path):

        for f in files:
            if f.endswith('.txt'):
                os.remove(os.path.join(root, f))
                # print(os.path.join(root, f))
    #         count += 1
    #         # print(os.path.join(root, f))
    #
    #     for d in dirs:
    #         print(os.path.join(root, d))
    # print(count)


def count_files(path):
    for root, dirs, files in os.walk(path):
        print(root, len(files))


def copy_file(src, dst):
    path, name = os.path.split(dst)
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copyfile(src, dst)


if __name__ == '__main__':
    walk_file('/nfs3/hjc/projects/RO/codes/tiny-imagenet')
