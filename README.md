Job2Resume
===========
<img width="998" alt="Screenshot 2024-05-22 at 23 36 43" src="https://github.com/Zhang-Zhaolong/Job2Resume/assets/70250233/4a9de0f1-66da-465c-9efe-a1d34af39cfd">

### Description
The source code of my undergraduate thesis `Job-Resume Matching Using Heterogeneous GNN` in 2022, advised by Proj. Jiawei Chen in SUFE. Above is the model architecture. 

### Environment 
```
Python                       3.8  
torch                        1.12.0+cu113
torch-geometric              2.0.4
torch-scatter                2.0.9
torch-sparse                 0.6.14
torchaudio                   0.12.0+cu113
torchvision                  0.13.0+cu113
dgl-cu113                    0.8.2.post1
sentence-transformers        2.2.0
```
### Usage
The origine dataset is job-resume text pairs, including job description and resume sentences. Since all the text is written by Chinese, so I use [bert-chinese](https://huggingface.co/google-bert/bert-base-chinese) as my pre-trained model. After the natrual language text are processed into embedding vectors, they will be put into a RGCN model to make predictions of whether a job and a resume can be matched.
