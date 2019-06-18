# CS385-Project1

This is an implementation for the first project of *CS385 Machine Learning* at Shanghai Jiao Tong University, instructed by Prof. Quanshi Zhang and Dr. Xu Cheng. I implemented some linear, kernel-based models as well as CNNs to do face classification, and also a sliding-window-based face detector. 

## Important Dependencies

The major dependencies of this project include:
- Python 3.6
- Numpy 1.14.0 +
- Scipy 0.19.0 +
- Sklearn
- Skimage
- h5py
- cv2
- **ThunderSVM** (The GPU version of SVM library, please refer to official document for installation)
- PyTorch 1.0 +

Notice: The SVMs are trained with **ThunderSVM** library, which is a little bit difficult to install. You may change my code in `main.py` to use a CPU version of SVM with `sklearn` or directly contact me.

## Dataset

You should download the FDDB dataset from [here](https://http://tamaraberg.com/faceDataset/originalPics.tar.gz)(579 M) and the annotations from [here](http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz)(161 K) and place them in the `data` folder under the root folder.

Then, you should extract the files with 
```
tar -zxvf originalPics.tar.gz
tar -zxvf FDDB-filds.tgz
```

After doing that, you should make sure that the `data` folder looks like:
```
- data
---- fddb
-------- 2002
-------- 2003
-------- FDDB-folds
------------ FDDB-fold-01-ellipseList.txt
```

Then, please go to the `data_processing` folder. There's a file `gen_bbox.py`. You should modify the `base_dir` global parameter on line 9 to adapt to your own computer. After that please run:
```
python gen_bbox.py
```

We will automatically generate hdf5 datasets for training/evaluation. Please prepare around 800 MB disk space to hold the generated hdf5 files.

## Evaluation

You can run `main.py` to evaluate our results. There are several flags in this file:
- model: The model to use. Please choose between "logistic", "svm", "lda" and "cnn".
- svm: The SVM kernel to use. Enabled only when model = "svm". Please choose between "linear", "rbf" and "poly".
- detection: Whether to run detection. Please choose between "True" and "False".
- vishog: Whether to visualize HOG features. Please choose between "True" and "False".
- vissv: Whether to visualize supporting vectors. Enabled only when model == "svm". Please choose between "True" and "False".
- train: Whether to train the model. For SVMs, you must train the model; for others, we have prepared the pretrained checkpoint under the root folder. Please choose between "True" and "False".

A sample code for running:
```
python main.py --model logistic --detection False --vishog True --vissv False --train False
```

You may modify it as you wish. If you choose `detection = True`, the results will be saved under the root folder.

## Additional Visualizations

You may run `python visualize.py` to visualize the distribution of HOG features (PCA/t-SNE). You can also run `python visualize_cnn.py` to get the t-SNE visualization for CNN features.

## Contact

If you have any questions, please contace me through email: kentang AT sjtu DOT edu DOT cn.
