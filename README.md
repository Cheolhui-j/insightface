# insightface

The pytorch implementation of [insightface](https://github.com/deepinsight/insightface)

![arcface](./image/arcface.png)

The following picture is based on [arcface](https://arxiv.org/pdf/1801.07698.pdf).

# Method

  The method is the same as [insightface](https://github.com/deepinsight/insightface).


# How to use 

+ Prepare Dataset

  Download & preprocess dataset

  It receives data in the existing rec format from [insightface](https://github.com/deepinsight/insightface) dataset-zoo.

  rec data is decompressed to make it into jpg format so that data in rec format can be used in pytorch.

  ```
  python prepare_data.py 

  usage: main.py  [--r REC_DATA_PATH]

  optional arguments:
        --rec_data_path REC_DATA_PATH Path where rec data file is stored  
  ```
        
+ Train
  
  After modifying each value in config.py, run the following command to learn.
  
  ```
  python train.py
  ```
  
+ Test
  
  To test the learned best model against lfw, run the following command to learn.
  
  ```
  python all_pair_test.py
  ```


# Results

# To Do
+ modify code 
+ add image

# Reference
Jiankang Deng and Jia Guo and Stefanos Zafeiriou(2018), ArcFace: Additive Angular Margin Loss for Deep Face Recognition

[paper](https://arxiv.org/pdf/1801.07698.pdf)  [code](https://github.com/deepinsight/insightface)
