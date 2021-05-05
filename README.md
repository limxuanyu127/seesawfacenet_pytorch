# Research project on lightweight Face Recognition model, using Seesafacenet

------

## 1. Intro

- This repo is a reimplementation of seesawfacenet[(paper)](https://arxiv.org/abs/1908.09124)
- The original github repo  is here: [(original repo)](https://github.com/cvtower/seesawfacenet_pytorch)

### **Objectives:**

- To identify new ways to make to improve on the chosen baseline model, which is the

------

## 2. Methdology

We explored new ways to make the Seesawfacenet faster and more accurate. The new changes that we succesfully incorporated are as follows:

- Replace the original Arc-Face loss function with the LiArc-Face loss function 
- Replace a Seesaw block with a Slim-CNN block 

We then evaluated the performance of our model compared to the original baseline Seesawfacenet. 

The literature review used are as follows:
- [(paper for LiArc-Face)](https://arxiv.org/pdf/1907.12256.pdf)
- [(paper for Slim-CNN)](https://arxiv.org/pdf/1907.02157.pdf)
- [(original repo for Slim-CNN)](https://github.com/gtamba/pytorch-slim-cnn)


## 3. How to use

Note that the orginal repo did not run smoothly for me, because of some errors, including dependency conflicts, wrong imports within certain files, and certain code segments that threw errors. Thus, I made some changes to the set-up steps. Here is how i set up the repo:

### 3.1 Setting up the environment

Python version 3.6 should be used. 

There is a *requirements.txt* file in the current repo, which can be used to install relevant dependencies. However, some dependencies are incompatible, so if you face issues after installing the dependencies in the files, do consider the following additional steps that I took:

  * To install a numpy version that is compatible with other dependencies:
    * pip install --upgrade numpy==1.16.0
  * To install other dependencies required for cv2:
    * sudo apt-get install libsm6 libxrender1 libfontconfig1 libxext6

### 3.2 Training the model

#### 3.2.1 Download the dataset

Download the MS1MV2 dataset:

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ), [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- More Dataset please refer to the [original post](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

**Note:** If you use [MS1MV2](https://arxiv.org/abs/1607.08221) dataset and the cropped [VGG2](https://arxiv.org/abs/1710.08092) dataset, please cite the original papers.

- after unzip the files to 'data' path, run :

  ```
  python prepare_data.py
  ```

  after the execution, you should find following structure:

```
faces_emore/
            ---> agedb_30
            ---> calfw
            ---> cfp_ff
            --->  cfp_fp
            ---> cfp_fp
            ---> cplfw
            --->imgs
            ---> lfw
            ---> vgg2_fp
```

#### 3.2.2  Training:

```
​```
python train.py -b [batch_size] -lr [learning rate] -e [epochs]

# python train.py -net mobilefacenet -b 256 -w 24
​```
```


### 3.3 Evaluating the model:

This repo, as with the original, helps you to evaluate your own photos. To do so:

#### 3.3.1 Prepare Facebank 

The Facebank is a consolidation of possible people that you want to detect. For example, if you want the code to be able to predict a person called Ben, provide face images of Ben in the data/face_bank folder, and guarantee it have a structure like following:

```
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```

Replace the id tags with the names of the person (eg. replace id1 with Ben).

#### 3.3.2 Prepare Test Image:
 In the directory *src/data/*, add the file you want to evaluate called *test_image.jpg*

#### 3.3.3 Run the evaluation:

Open the predict.py file, and change the file path for the trained model. 

```
python new_predict.py --path data/test_image.jpg  
#Optional -u arg to update the facebank, included for first use or when u add new imges to the facebank

```

The outputs should be saved in *data/save/*

#### Note:
For other capabilities of the repo, do refer to the Readme in the original repo.


## 4. References

- This repo is mainly based on [cvtower/SeesawNet_pytorch](https://github.com/cvtower/SeesawNet_pytorch).

- It is also inspired by the following works:
    * [(paper for LiArc-Face)](https://arxiv.org/pdf/1907.12256.pdf)
    * [(paper for Slim-CNN)](https://arxiv.org/pdf/1907.02157.pdf)
    * [(original repo for Slim-CNN)](https://github.com/gtamba/pytorch-slim-cnn)
