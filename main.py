###########Face Detection With Linear Models###########
# Author: Haotian Tang                                                      # 
# E-Mail: kentang@sjtu.edu.cn                                            #
# Date: May, 2019                                                              #
#################################################

import argparse
import os
import h5py
import numpy as np
import cv2
from scipy.stats import entropy
from sklearn.externals import joblib
from methods.logistic_regression import LogisticRegression
from methods.lda import LinearDiscriminantAnalysis
from methods.cnn import CNN

from thundersvm import SVC
from sklearn.svm import SVC as SVCCPU
from data_processing.gen_bbox import crop_face, base_dir, \
    visualize_face, hog_visualize
from skimage.feature import hog


# Detection pipeline
# 0. Define initial trial box size to be 200, 150, 100.
# 1. Sliding window, using fixed box size, fixed stride, to generate candidates.
# 2. Pass all candidates through classifier, remain positive samples
# 3. Do NMS.
# 4. At positive positions, enumerate bbox size.
# 5. Pass all candidates through classifier, remain positive samples
# 6. Do NMS again.
# 7. Filter out unreliable predictions by color entropy suppression. (HOG: no RGB!)
# 8. If still some detections are found, end. Else, reduce initial box size, try again.
# Update: I believe we can make the "fixed box size" in 1 not fixed by enumerating over multiple choices.

def sliding_window(img_fn, box_size, stride):
    # box_size: (int, int) tuple Box size to try.
    # stride: (int, int) tuple. Stride at both directions.
    fn = os.path.join(base_dir, img_fn)
    im = cv2.imread(fn)
    print(im.shape)
    xsize, ysize = box_size
    xstride, ystride = stride
    cands = []
    for x in range(0, im.shape[0], xstride):
        for y in range(0, im.shape[1], ystride):
            cropped_cand = crop_face(im, x, y, xsize, ysize)
            resized_cand = cv2.resize(cropped_cand, (96,96))
            cands.append((x, y, xsize, ysize, resized_cand))
    return cands



# calculate the overlap between two bboxes. The returned value is IoU.
def calc_overlap(bbox1, bbox2):
    x1min, y1min, x1s, y1s = bbox1
    x2min, y2min, x2s, y2s = bbox2
    if not (x2min < x1min + x1s and y2min < y1min + y1s) or not (x1min < x2min + x2s and y1min < y2min + y2s):
        return 0
    
    # need to consider: bbox2 entirely in bbox1 or reverse?
    if x2min < x1min + x1s and y2min < y1min + y1s:
        overlap_size = (x1min + x1s - x2min) * (y1min + y1s - y2min) if x2min + x2s > x1min + x1s else x2s * y2s
    else:
        overlap_size = (x2min + x2s - x1min) * (y2min + y2s - y1min) if x1min + x1s > x2min + x2s else x1s * x2s
    
    total_size = x1s * y1s + x2s * y2s - overlap_size
    iou = float(overlap_size) / total_size
    return iou

# generate a resized bounding box
def resize_bbox(bbox, ratio):
    # bbox: 4-tuple: xmin, ymin, xsize, ysize
    # ratio: 2-tuple, xratio, yratio
    xmin, ymin, xsize, ysize = bbox
    xratio, yratio = ratio
    xcenter, ycenter = xmin + xsize // 2, ymin + ysize // 2
    x_newsize, y_newsize = int(xsize * xratio), int(ysize * yratio)
    x_newmin, y_newmin = xcenter - x_newsize //2, ycenter - y_newsize // 2
    return (x_newmin, y_newmin, x_newsize, y_newsize)


# classify a single image as face / non-face.
def judge_single(model, im):
    model_name = model.__class__.__name__
    if 'CNN' not in model_name:
        hog_feature = hog(im, 9, (16,16), (2,2)).reshape(1, -1)
    
    # To be reformatted.
    if 'CNN' in model_name:
        return model.predict(im[None,:,:,:].transpose(0,3,1,2), return_prob=True)
    elif 'Logistic' in model_name:
        return model.predict_proba(hog_feature)[0]
    else:
        return model.decision_function(hog_feature)[0]


# detection based on classifier for a candidate list.
def detect_list(model, cand_lis):
    model_name = model.__class__.__name__
    positives = []
    for cand in cand_lis:
        x, y, xsize, ysize, resized_cand = cand
        cls = judge_single(model, resized_cand)
        
        # to be reformatted.
        # logistic
        if 'CNN' in model_name:
            prob = cls.item()
            if prob > 0.5:
                positives.append((x, y, xsize, ysize, prob))
        
        elif "Logistic" in model_name:
            neg_prob, pos_prob = cls[0], cls[1]
            if pos_prob > 0.5:
                positives.append((x, y, xsize, ysize, pos_prob))
        
        else:
            prob = cls[0]
            if prob > 0.:
                positives.append((x, y, xsize, ysize, prob))
            
    return positives


# non-maximal suppression
# sort all bboxes by confidence. Go from high to low. Keep a bbox if
# overlap with previous ones below threshold. I think its O(n^2). to be optimized.
def nms(pos_lis, threshold):
    sorted_pos_lis = pos_lis.copy()
    sorted_pos_lis.sort(key=lambda a:a[-1], reverse=True)
    suppressed_lis = []
    for i in range(len(sorted_pos_lis)):
        accept = True
        for j in range(i):
            iou = calc_overlap(sorted_pos_lis[i][:-1], sorted_pos_lis[j][:-1])
            if iou >= threshold:
                accept = False
                break
        
        if accept:
            suppressed_lis.append(sorted_pos_lis[i])
    
    return suppressed_lis

# filter out unreliable detection results using color entropy.
def color_entropy_filtering(cand_lis, threshold):
    cand_idx_lis = []
    final_cand_lis = []
    for i in range(len(cand_lis)):
        cand = cand_lis[i]
        color_histogram = np.bincount(cand.reshape(-1), minlength=256).astype(np.float32)
        color_histogram /= color_histogram.sum()
        cand_entropy = entropy(color_histogram)
        if cand_entropy >= threshold:
            final_cand_lis.append(cand)
            cand_idx_lis.append(i)
            print(i, cand_entropy, threshold)
    return final_cand_lis, cand_idx_lis


def detect(model, img_fn):
    # enumerate from large initial sizes to smaller ones.
    initial_sizes = [150,100]
    
    for sz in initial_sizes:
        xratios = [0.5, 0.75, 1, 1.25, 1.5]
        yratios = [0.5, 0.75, 1, 1.25, 1.5]
        ratios = []
        for xr in xratios:
            for yr in yratios:
                if xr >= yr:
                    ratios.append((xr, yr))

        sliding_windows = sliding_window(img_fn, (sz,sz), (10, 10))
        positive_cands1 = detect_list(model, sliding_windows)
        nms_cands1 = nms(positive_cands1, 0.5)

        print(nms_cands1, len(nms_cands1))

        cropped_lis = []
        for cand in nms_cands1:
            cropped_lis.append(cv2.resize(crop_face(img_fn, *cand[:-1]), (96,96)))
        cropped_lis = np.array(cropped_lis)
        cropped_lis = cropped_lis.reshape(-1, *cropped_lis.shape[2:]).astype(np.uint8)
        cv2.imwrite('test.png', cropped_lis)

        new_cands = []
        for cand in nms_cands1:
            for rat in ratios:
                new_bbox = resize_bbox(cand[:-1], rat)
                new_cropped = cv2.resize(crop_face(img_fn, *new_bbox), (96,96))
                new_cands.append((*new_bbox, new_cropped))
        positive_cands2 = detect_list(model, new_cands)
        nms_cands2 = nms(positive_cands2, 0.5)

        print(nms_cands2, len(nms_cands2))

        # if the highest confidence is too low, we continue.
        if nms_cands2[0][-1] < 0.7:
            continue
        
        cropped_lis = []
        for cand in nms_cands2:
            cropped_lis.append(cv2.resize(crop_face(img_fn, *cand[:-1]), (96,96)))

        # color entropy filtering
        cropped_lis, cand_idx_lis = color_entropy_filtering(cropped_lis, 5.0)
        
        # no result, continue
        if len(cropped_lis) == 0:
            continue
        
        cropped_lis = np.array(cropped_lis)
        cropped_lis = cropped_lis.reshape(-1, *cropped_lis.shape[2:]).astype(np.uint8)

        cv2.imwrite('test1.png', cropped_lis)

        return [tuple(nms_cands2[i][:-1]) for i in cand_idx_lis]

    
def loading_data():
    # For HOG-based linear models.
    
    try:
        db = h5py.File('data_processing/fddb_hog_train.h5')
    except:
        print("h5py database is non-existent! Please run data processing first.")
        return
    
    train_data = db['data'][...]
    train_label = db['label'][...].astype(np.int32)
    db.close()
    
    idx = np.arange(train_data.shape[0])
    np.random.shuffle(idx)
    train_data = train_data[idx, :]
    train_label = train_label[idx]


    db = h5py.File('data_processing/fddb_hog_test.h5')
    test_data = db['data'][...]
    test_label = db['label'][...].astype(np.int32)
    db.close()
    
    # CNN data
    
    cnn_train_data = []
    cnn_train_label = []
    cnn_test_data = []
    cnn_test_label = []
    # positive train
    for fold in range(8):
        db = h5py.File('data_processing/fddb_positive_%d.h5'%(fold+1), 'r')
        cur_data = db['data'][...]
        cnn_train_data.append(cur_data.transpose(0,3,1,2))
        cnn_train_label.append(np.ones(cur_data.shape[0]))
        db.close()
    
    # positive test
    for fold in range(8,10):
        db = h5py.File('data_processing/fddb_positive_%d.h5'%(fold+1), 'r')
        cur_data = db['data'][...]
        cnn_test_data.append(cur_data.transpose(0,3,1,2))
        cnn_test_label.append(np.ones(cur_data.shape[0]))
        db.close()
    
    
    # negative train
    for fold in range(4):
        db = h5py.File('data_processing/fddb_negative_%d.h5'%(fold+1), 'r')
        cur_data = db['data'][...]
        cnn_train_data.append(cur_data.transpose(0,3,1,2))
        cnn_train_label.append(np.zeros(cur_data.shape[0]))
        db.close()
    
    # negative_test
    for fold in [9]:
        db = h5py.File('data_processing/fddb_negative_%d.h5'%(fold+1), 'r')
        cur_data = db['data'][...]
        cnn_test_data.append(cur_data.transpose(0,3,1,2))
        cnn_test_label.append(np.zeros(cur_data.shape[0]))
        db.close()
    
    cnn_train_data = np.vstack(cnn_train_data)
    cnn_train_label = np.concatenate(cnn_train_label)
    cnn_test_data = np.vstack(cnn_test_data)
    cnn_test_label = np.concatenate(cnn_test_label)
    return train_data, train_label, test_data, test_label, cnn_train_data, \
        cnn_train_label, cnn_test_data, cnn_test_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face classification and detection with linear models.')

    parser.add_argument('--model', default='logistic', type=str, help='The model used for face classification.')
    parser.add_argument('--svm', default='linear', type=str, help='Type of SVM, enabled if we are using SVMs.')
    parser.add_argument('--detection', default='False', type=str, help='Whether to detect.')
    parser.add_argument('--vishog', default='False', type=str, help='Whether to visualize HOG features.')
    parser.add_argument('--vissv', default='False', type=str, help='Whether to visualize supporting vectors.')
    parser.add_argument('--train', default='True', type=str, help='Whether to train the model.')
    
    args = parser.parse_args()
    args.detection = False if args.detection.lower() != 'true' else True
    args.vishog = False if args.vishog.lower() != 'true' else True
    args.vissv = False if args.vissv.lower() != 'true' else True
    args.train = False if args.train.lower() != 'true' else True
    
    if args.model.lower() == 'svm':
        args.train = True
    
    print("Loading database......")
    ret = loading_data()
    if ret is not None:
        train_data, train_label, test_data, test_label, cnn_train_data, \
            cnn_train_label, cnn_test_data, cnn_test_label = ret
    else:
        exit(1)
    
    
    print("Creating model......")
    
    args.model = args.model.lower()
    args.svm = args.svm.lower()
    
    if args.model == 'logistic':
        # Logistic -> 0.9651
        model = LogisticRegression()
    elif args.model == 'lda':
        # LDA -> 0.9656
        model = LinearDiscriminantAnalysis()
    elif args.model == 'svm':
        
        if args.svm == 'linear':
            # SVM-linear -> 0.9658
            model = SVC(kernel='linear')
        elif args.svm == 'rbf':
            # SVM-rbf: gamma = 0.8 -> 0.9764
            model = SVC(kernel='rbf', gamma=0.8)
        elif args.svm == 'poly':
            # SVM-polynomial: degree = 3, gamma = 0.8 -> 0.9778
            model = SVC(kernel='polynomial', degree=3, gamma=0.8)
        else:
            raise NotImplementedError
    elif args.model == 'cnn':
        # CNN -> 0.9869
        model = CNN()
    else:
        raise NotImplementedError
    
    if args.train:
        print("Training %s......"%args.model)
        if args.model != 'cnn':
            model.fit(train_data, train_label)
        else:
            model.fit(cnn_train_data, cnn_train_label)
    
    print("Predicting with trained %s......"%args.model)
    if args.model != 'cnn':
        print(model.score(test_data, test_label))
    else:
        print(model.score(cnn_test_data, cnn_test_label))
    
    # Visualize Supporting Vectors.
    if args.model == 'svm' and args.vissv:
        print("Visualizing support vectors ......")
        sv_array = []
        rows = 3
        cols = 4
        sv = np.random.choice(model.support_, rows*cols)
        sv_samples = cnn_train_data[sv]
        sv_labels = cnn_train_label[sv]
        for i in range(rows):
            sv_array.append([])
            for j in range(cols):
                sample = sv_samples[i*cols+j].transpose(1,2,0)
                if sv_labels[i*cols+j] == 1:
                    sample = cv2.rectangle(sample, (0, 0), (93, 93), (0,255,0), 3)
                else:
                    sample = cv2.rectangle(sample, (0, 0), (93, 93), (0,0,255), 3)
                sv_array[-1].append(sample)
        sv_array = np.array(sv_array).astype(np.uint8)
        sv_array = sv_array.transpose(0,2,1,3,4).reshape(96*rows, 96*cols, 3)
        cv2.imwrite("sv.png", sv_array)
    
    
    # Visualize HOG features
    if args.vishog:
        hog_visualize()
    
    
    # Detection and Visualization
    if args.detection:
        print("Detecting......")
        fold_file = 'data/fddb/FDDB-folds/FDDB-fold-10.txt'
        with open(fold_file, 'r') as f:
            lines = f.readlines()
        lines.sort()
        candidates = lines
        model = LogisticRegression()
        for img_fn_ in candidates:
            img_fn = img_fn_.rstrip() + '.jpg'
            visualized = visualize_face(img_fn)
            if visualized is not None:
                cv2.imwrite(img_fn.split('/')[-1].replace('.jpg', '_bbox.jpg'), visualized)
                print("Visualized GT bounding boxes in %s."%img_fn)
            detected = detect(model, img_fn)
            visualized_ours = visualize_face(img_fn, detected)
            cv2.imwrite(img_fn.split('/')[-1].replace('.jpg', '_bbox_pred.jpg'), visualized_ours)
            print("Visualized predicted bounding boxes in %s."%img_fn)

