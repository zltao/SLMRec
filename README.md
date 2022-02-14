# SLMRec

"Self-supervised Learning for Multimedia" (SLMRec) aims capture multi-modal patterns in the data itself, we go beyond the supervised learning paradigm, and incorporate the idea of self-supervised learning (SSL) into multimedia recommendation.

> Authors: Zhulin Tao, Xiaohao Liu, Yewei Xia, Xiang Wang, Lifang Yang, Xianglin Huang, Tat-Seng Chua

<figure> < img src="figures/framework.png" height="400"></figure>

## Installation
The code has been tested running under Python 3.6.5. The required packages are as follows:
* torch==1.7.0
* numpy==1.16.1
* torch_geometric==1.6.1

## Data download
We provide three processed datasets: Kwai, Tiktok, and Movielnes.  
- You can find the full version of recommendation datasets via [Kwai](https://www.kuaishou.com/activity/uimc), [Tiktok](http://ai-lab-challenge.bytedance.com/tce/vc/), and [Movielens](https://grouplens.org/datasets/movielens/).
Since the copyright of datasets, we cannot release them directly. 

## Run SLMRec
The hyper-parameters used to train the models are set as default in the `conf/SLMRec.properties`. Feel free to change them if needed.

```sh
python3 main.py --recommender="SLMRec" --data.input.dataset=tiktok
```
