import numpy as np
import cv2
from glob import glob
import sys
import math
from demo import rot, get_bbox, retrive_bbox3d

class preprocessor_test:
    def __init__(self):
        return None

    def printProgress(self, iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50):
        """
        This is only a function to indicate progress and does nothing else
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        formatStr       = "{0:." + str(decimals) + "f}"
        percents        = formatStr.format(100 * (iteration / float(total)))
        filledLength    = int(round(barLength * iteration / float(total)))
        bar             = '|' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def generateAllData(self):
        self.bCount = 0
        #files = glob('../deploy/trainval/*/*_image.jpg')
        files = glob('deploy/test/*/*_image.jpg')
        self.flen = len(files)
        #self.flen = 10
        #self.imgs = np.zeros((self.flen, 526, 957, 4), dtype=np.float32)
        #self.labels = np.zeros((self.flen, 526, 957, 1), dtype=np.float32)
        namerec = open('train/imgs.txt', 'w')
        print('Loading all data')
        for i in range(self.flen):
            imgs = np.zeros((526, 957, 4), dtype=np.float32)
            snapshot = files[i]
            #print(snapshot)
            img = cv2.imread(snapshot).astype(np.float32)
            img = img/255.0
            imgs[:, :, :3] = cv2.resize(img, (957, 526))
            #dps = np.load(snapshot.replace('_image.jpg', '_depthImage.npy'))
            #self.imgs[i, :, :, 3] = cv2.resize(dps, (304, 224))
            #generate depth from lidar
            xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
            xyz.resize([3, xyz.size // 3])
            # get projection matrix
            proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
            proj.resize([3, 4])
            # project clound points onto image
            uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
            uv = uv / uv[2, :]
            clr = np.linalg.norm(xyz, axis=0)
            dps = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            dpsmk = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            # https://en.wikipedia.org/wiki/Bilateral_filter
            for uvidx in range(uv.shape[1]):
                if int(uv[0, uvidx]) >= 0 and int(uv[0, uvidx]) <= dps.shape[1] and int(uv[1, uvidx]) >= 0 and int(uv[1, uvidx]) <= dps.shape[0]:
                    dps[ int(uv[1, uvidx]), int(uv[0, uvidx]) ] = clr[uvidx]
                    dpsmk[ int(uv[1, uvidx]), int(uv[0, uvidx]) ] = 1
            ksize = 11
            dps_blurred = cv2.GaussianBlur(dps, (ksize, ksize), 0)
            weight_count = cv2.GaussianBlur(dpsmk, (ksize, ksize), 0)
            weight_count += 1e-9
            res = np.divide(dps_blurred, weight_count)

            skyline_thresh = 500

            res1 = res[:skyline_thresh, :]
            res2 = res[skyline_thresh:, :]
            res1[np.where(res1<=5)] = 60
            res2[np.where(res2<=5)] = 5
            res[:skyline_thresh, :] = res1
            res[skyline_thresh:, :] = res2

            #res[np.where(res<=5)] = 100

            #res[np.where(res<=5)] = 100
            res_blurred = cv2.GaussianBlur(res, (ksize, ksize), 0)
            rm = np.amax(res_blurred)
            res_blurred = 1-res_blurred/rm
            imgs[:, :, 3] = cv2.resize(res_blurred, (957, 526))

            target_img = np.zeros((526, 957, 3), dtype=np.float32)
            target_img[:, :, 0] = np.add(imgs[:,:,0], imgs[:,:,1])/2
            target_img[:, :, 1] = np.add(imgs[:,:,1], imgs[:,:,2])/2
            target_img[:, :, 2] = imgs[:,:,3]

            #cli = np.load(snapshot.replace('_image.jpg', '_carLabelImage.npy'))
            #generate car boxes from bbox
            #spt = snapshot.split('trainval\\')
            spt = snapshot.split('test\\')
            sps = spt[1].split('\\')
            uuid = sps[0]
            imids = sps[1].split('_')
            imid = imids[0]

            #save_path = '../../darknet/build/darknet/x64/test3/data/' + uuid + '_' + imid
            save_path = 'train/data/' + uuid + '_' + imid
            cv2.imwrite(save_path + '.jpg', target_img*255)
            namerec.write('data/' + uuid + '_' + imid + '.jpg' + '\n')

            self.printProgress(i, self.flen)
        print('generate data okay')
