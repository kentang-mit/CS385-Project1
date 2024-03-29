import sys
import os
import h5py
import numpy as np
import cv2
import pickle
from skimage.feature import hog

base_dir = '/home/tang/machine-learning/CS385-project/data/fddb/'

# TBD consider bottom right corner padding!
def read_bbox_one_fold(fold_idx, fold_file):
    
    f = open(fold_file, 'r')
    lines = f.readlines()
    f.close()
    cnt = 0
    all_annos = []
    all_samples = []
    while cnt < len(lines):
        faces_anno = []
        img_fn = lines[cnt].rstrip()
        cnt += 1
        num_faces = int(lines[cnt])
        cnt += 1
        
        for n in range(num_faces):
            cur_anno = [float(x) for x in lines[cnt].split()]
            # Discard the angle and the binary label.
            main_axis, minor_axis, _, ycenter, xcenter, _ = tuple(cur_anno)
            xmin = int(xcenter - main_axis * 4 / 3)
            ymin = int(ycenter - minor_axis * 4 / 3)
            main_size = int(main_axis * 4 / 3) * 2
            minor_size = int(minor_axis * 4 / 3) * 2
            
            faces_anno.append(np.array([xmin, ymin, main_size, minor_size]))
            cropped_face = crop_face(img_fn, xmin, ymin, main_size, minor_size)
            cropped_face = cv2.resize(cropped_face, (96,96))
            all_samples.append(cropped_face)
            cnt += 1
        all_annos.append(faces_anno)
    
    all_samples = np.array(all_samples).astype(np.uint8)
    dbfile = h5py.File('fddb_positive_%d.h5'%fold_idx,'w')
    samples = dbfile.create_dataset('data', data=all_samples, compression="gzip")
    dbfile.close()
    f = open('fddb_positive_anno_%d.h5'%fold_idx,'wb')
    pickle.dump(all_annos, f)
    f.close()

def crop_face(img_fn, xmin, ymin, main_size, minor_size):
    # Crop the bounding box from an image with given bounding box.
    # Read the image
    if type(img_fn) == str:
        new_img_fn = img_fn + '.jpg' if '.jpg' not in img_fn else img_fn
        im_ = cv2.imread(os.path.join(base_dir, new_img_fn))
    else:
        im_ = img_fn
    
    xmax, ymax, _ = im_.shape
    xmin, ymin, main_size, minor_size = int(xmin), \
        int(ymin), int(main_size), int(minor_size)
    
    xpad, ypad = 0, 0
    # Padding calculation for dim 0.
    if xmin < 0:
        xpad = -xmin
    if xmin + main_size >= xmax:
        xpad = max(xpad, xmin + main_size + 1 - xmax)
    
    # Padding calculation for dim 1.
    if ymin < 0:
        ypad = -ymin
    if ymin + minor_size >= ymax:
        ypad = max(ypad, ymin + minor_size + 1 - ymax)
    
    xmin += xpad
    ymin += ypad
    xmax += 2 * xpad
    ymax += 2 * ypad
    
    # Padding if necessary
    if xpad == 0 and ypad == 0:
        im = im_
    else:
        im = np.zeros((im_.shape[0]+2*xpad, im_.shape[1]+2*ypad, 3))
        
        # nearest neighborhood padding
        # center
        im[xpad:xmax-xpad, ypad:ymax-ypad, :] = im_
        # pad top
        im[:xpad, ypad:ymax-ypad, :] = im_[0, :, :][None, :, :]
        # pad left
        im[xpad:xmax-xpad, :ypad, :] = im_[:, 0, :][:, None, :]
        # pad upper-left
        im[:xpad, :ypad, :] = im_[0, 0, :][None, None, :]
        # pad lower-left
        im[xmax-xpad:, :ypad, :] = im_[-1, 0, :][None, None, :]
        
        # pad down
        im[xmax-xpad:, ypad:ymax-ypad, :] = im_[-1, :, :][None, :, :]
        # pad right
        im[xpad:xmax-xpad, ymax-ypad:, :] = im_[:, -1, :][:, None, :]
        # pad lower-right
        im[xmax-xpad:, ymax-ypad:, :] = im_[-1, -1, :][None, None, :]
        # pad upper-right
        im[:xpad, ymax-ypad:, :] = im_[0, -1, :][None, None, :]
        
        
        im = im.astype(np.uint8)
    
    # Debug: Print the bounding box information.
    #print(xmin, ymin, xmin+minor_size, ymin+main_size, xmax, ymax)
    
    # Cropping use the bounding box.
    cropped_im = im[xmin:xmin+main_size, ymin:ymin+minor_size, :]
    return cropped_im


def gen_negative_one_fold(fold_idx, fold_file):
    f = open(fold_file, 'r')
    lines = f.readlines()
    f.close()
    cnt = 0
    all_samples = []
    while cnt < len(lines):
        faces_anno = []
        img_fn = lines[cnt].rstrip()
        cnt += 1
        num_faces = int(lines[cnt])
        cnt += 1
        for n in range(num_faces):
            cur_anno = [float(x) for x in lines[cnt].split()]
            # Discard the angle and the binary label.
            main_axis, minor_axis, _, ycenter, xcenter, _ = tuple(cur_anno)
            
            xmin = int(xcenter - main_axis)
            ymin = int(ycenter - minor_axis)
            main_size = int(main_axis * 2)
            minor_size = int(minor_axis * 2)
            
            # eight shifted bounding boxes + 1 resized global image
            global_im = cv2.imread(os.path.join(base_dir, img_fn+'.jpg'))
            global_im = cv2.resize(global_im, (96,96))
            
            shift_sizes = ((-1./3, -1./3), (-1./3, 0), (-1./3, 1./3), (0, -1./3), (0, 1./3), (1./3, -1./3), (1./3, 0.), (1./3, 1./3))
            neg_samples = []
            cropped_faces = []
            idx = 0
            for shift_size in shift_sizes:
                idx += 1
                xratio, yratio = shift_size
                shift_x = xratio * main_size
                shift_y = yratio * minor_size
                cur_xmin = xmin + shift_x
                cur_ymin = ymin + shift_y
                neg_samples.append(np.array([cur_xmin, cur_ymin, main_size, minor_size]))
                cropped_face = crop_face(img_fn, cur_xmin, cur_ymin, main_size, minor_size)
                cropped_face = cv2.resize(cropped_face, (96,96))
                cropped_faces.append(cropped_face)
            
            cropped_faces.append(global_im)
            all_samples.append(np.array(cropped_faces))
            cv2.imwrite('test1.png', np.stack(cropped_faces, 1).reshape(cropped_faces[0].shape[0],-1,3))
            
            cnt += 1
    
    all_samples = np.vstack(all_samples).astype(np.uint8)
    dbfile = h5py.File('fddb_negative_%d.h5'%fold_idx,'w')
    samples = dbfile.create_dataset('data', data=all_samples, compression="gzip")
    dbfile.close()
    
def extract_hog(positive_idx, negative_idx, split="train"):
    # Extract hog feature from database file and save to another hdf5 dataset.
    labels = []
    samples = []
    for idx in positive_idx:
        print("Generating HOG feature for fold %d (positive images) in FDDB......"%idx)
        dbfile = 'fddb_positive_%d.h5'%idx
        dbfile = h5py.File(dbfile,'r')
        data = dbfile['data'][...]
        for i in range(len(data)):
            hog_feature = hog(data[i], 9, (16,16), (2,2))
            samples.append(hog_feature)
            labels.append(1)
    
    for n_idx in negative_idx:
        print("Generating HOG feature for fold %d (negative images) in FDDB......"%n_idx)
        dbfile = 'fddb_negative_%d.h5'%n_idx
        dbfile = h5py.File(dbfile,'r')
        data = dbfile['data'][...]
        for i in range(len(data)):
            hog_feature = hog(data[i], 9, (16,16), (2,2))
            samples.append(hog_feature)
            labels.append(-1)
    
    samples = np.array(samples)
    labels = np.array(labels)
    dbfile = "fddb_hog_%s.h5"%split
    dbfile = h5py.File(dbfile, 'w')
    data = dbfile.create_dataset('data', data=samples, compression="gzip")
    label = dbfile.create_dataset('label', data=labels, compression="gzip")
    dbfile.close()


# Visualize the bounding box on an image.
# img should be a numpy array and bbox in (xmin, ymin, xsize, ysize) format.
# TODO: remove overlapping code with crop_face
def visualize_bbox(im_, bbox, color=(255,0,0)):
    xmin, ymin, main_size, minor_size = bbox
    
    xmax, ymax, _ = im_.shape
    xmin, ymin, main_size, minor_size = int(xmin), \
        int(ymin), int(main_size), int(minor_size)
    
    xpad, ypad = 0, 0
    # Padding calculation for dim 0.
    if xmin < 0:
        xpad = -xmin
    if xmin + main_size >= xmax:
        xpad = max(xpad, xmin + main_size + 1 - xmax)
    
    # Padding calculation for dim 1.
    if ymin < 0:
        ypad = -ymin
    if ymin + minor_size >= ymax:
        ypad = max(ypad, ymin + minor_size + 1 - ymax)
    
    xmin += xpad
    ymin += ypad
    xmax += 2 * xpad
    ymax += 2 * ypad
    
    # Padding if necessary
    if xpad == 0 and ypad == 0:
        im = im_
    else:
        im = np.zeros((im_.shape[0]+2*xpad, im_.shape[1]+2*ypad, 3))
        
        # nearest neighborhood padding
        # center
        im[xpad:xmax-xpad, ypad:ymax-ypad, :] = im_
        # pad top
        im[:xpad, ypad:ymax-ypad, :] = 0#im_[0, :, :][None, :, :]
        # pad left
        im[xpad:xmax-xpad, :ypad, :] = 0#im_[:, 0, :][:, None, :]
        # pad upper-left
        im[:xpad, :ypad, :] = 0#im_[0, 0, :][None, None, :]
        # pad lower-left
        im[xmax-xpad:, :ypad, :] = 0#im_[-1, 0, :][None, None, :]
        
        # pad down
        im[xmax-xpad:, ypad:ymax-ypad, :] = 0#im_[-1, :, :][None, :, :]
        # pad right
        im[xpad:xmax-xpad, ymax-ypad:, :] = 0#im_[:, -1, :][:, None, :]
        # pad lower-right
        im[xmax-xpad:, ymax-ypad:, :] = 0#im_[-1, -1, :][None, None, :]
        # pad upper-right
        im[:xpad, ymax-ypad:, :] = 0#im_[0, -1, :][None, None, :]
        
        
        im = im.astype(np.uint8)
    
    cv2.rectangle(im, (ymin, xmin), (ymin+minor_size, xmin+main_size), color, 3)
    return im, xpad, ypad


# Visualize the bbox on an image. Will call visualize_bbox
# bboxes: list of bbox
def visualize_face(img_fn, bboxes=None, wrong=False):
    # If bbox is known
    im = cv2.imread(os.path.join(base_dir, img_fn))
    if bboxes is not None:
        for bbox in bboxes:
            im, _, _ = visualize_bbox(im, bbox)
        return im
    
    # Visualize GT bbox and negative bbox.
    if not os.path.exists("img_idx_correspondence.pkl"):
        dic = {}
        for i in range(10):
            cnt = 0
            f = open(os.path.join(base_dir, "FDDB-folds/FDDB-fold-%02d.txt"%(i+1)), 'r')
            files = [x.rstrip()+'.jpg' for x in f.readlines()]
            f.close()
            for file in files:
                dic[file] = (i+1, cnt)
                cnt += 1
        f = open("img_idx_correspondence.pkl", "wb")
        pickle.dump(dic, f)
        f.close()
    
    f = open("img_idx_correspondence.pkl", "rb")
    correspondence = pickle.load(f)
    f.close()
    
    if img_fn not in correspondence.keys():
        print("%s is not annotated, please detect the face first."%img_fn)
        return
   
    fold, idx = correspondence[img_fn]
    with open("data_processing/fddb_positive_anno_%d.pkl"%fold, "rb") as f:
        annos = pickle.load(f)
    
    tot_xpad, tot_ypad = 0, 0
    cur_anno = annos[idx]
    
    # negative
    
    shift_sizes = ((1./3, -1./3), (0, 1./3), (0, -1./3), (1./3, 1./3), (1./3, 0), (-1./3,1./3), (-1./3, -1./3), (-1./3, 0))
    for anno in cur_anno:
        xmin, ymin, main_size, minor_size = anno
        main_size *= 3./4
        minor_size *= 3./4
        for shift_size in shift_sizes:
            xratio, yratio = shift_size
            shift_x = xratio * main_size
            shift_y = yratio * minor_size
            cur_xmin = xmin + shift_x
            cur_ymin = ymin + shift_y
            """
            if cur_xmin < 0:
                main_size += cur_xmin
                cur_xmin = 0
            if cur_ymin < 0:
                minor_size += cur_ymin
                cur_ymin = 0
            """
            im, xpad, ypad = visualize_bbox(im, (cur_xmin+tot_xpad, cur_ymin+tot_ypad, main_size, minor_size), color=(0, 0, 255))
            tot_xpad += xpad
            tot_ypad += ypad
        
        # positive
        for anno in cur_anno:
            im, xpad, ypad = visualize_bbox(im, (anno[0]+tot_xpad, anno[1]+tot_ypad, *anno[2:]))
            
            tot_xpad += xpad
            tot_ypad += ypad
            
            im, xpad, ypad = visualize_bbox(im, (anno[0]+tot_xpad, anno[1]+tot_ypad, anno[2] * 0.75, anno[3] * 0.75), color=(0, 255, 0))
    
    return im
    

def hog_visualize(fold=None, rows=4, cols=3):
    if fold is None:
        fold = 1
    
    hog_h5 = h5py.File('data_processing/fddb_positive_%d.h5'%fold,'r')
    data = hog_h5['data'][...][:rows*cols, ...]
    hogs = [hog(data[i,...], 9, (16,16), (2,2), visualize=True)[1] for i in range(len(data))]
    data = data.reshape(-1, 96, 3)
    hogs = np.vstack(hogs)
    hogs = (hogs - hogs.min()) / (hogs.max()-hogs.min()) * 255.
    hogs = np.tile(hogs[:,:,None], [1,1,3])
    img = np.hstack([data, hogs]).reshape(-1, 96, 192, 3)
    new_img = []
    for r in range(rows):
        new_img.append(np.hstack(img[r * cols: (r+1) * cols]))
    new_img = np.vstack(new_img)
    # reshape the image
    cv2.imwrite("hogs.png", new_img)
    
if __name__ == '__main__':
    # Generate positive images for face classification
    for i in range(10):
        #if os.path.exists('fddb_positive_%d.h5'%(i+1)):
        #    continue
        print("Generating positive images for fold %d in FDDB......"%(i+1))
        read_bbox_one_fold(i+1, os.path.join(base_dir, 'FDDB-folds/FDDB-fold-%02d-ellipseList.txt'%(i+1)))
    
    # Generate negative images for face classification
    for i in [0,1,2,3,9]:
        #if os.path.exists('fddb_negative_%d.h5'%(i+1)):
        #    continue
        print("Generating negative images for fold %d in FDDB......"%(i+1))
        gen_negative_one_fold(i+1, os.path.join(base_dir, 'FDDB-folds/FDDB-fold-%02d-ellipseList.txt'%(i+1)))
    
    # Extract HOG features for training set.
    extract_hog(list(range(1,9)), [1,2,3,4], "train")
    
    # Extract HOG features for testing set.
    extract_hog([9,10], [10], "test")
    