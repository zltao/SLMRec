# Self-supervised  learning for Multimedia Recommendation
This is our Pytorch implementation for the paper:  
> Zhulin Tao, Xiaohao Liu, Yewei Xia, Xiang Wang*, Lifang Yang, Xianglin Huang, and Tat-Seng Chua. Self-supervised  learning for Multimedia Recommendation  
Author: Dr.Zhulin Tao (taozhulin at gmail.com)

## Introduction
Learning representations for multimedia content is critical for multimedia recommendation. go beyond the supervised learning paradigm, and incorporate the idea of self-supervised learning (SSL) into multimedia recommendation.

## Citation
If you want to use our codes and datasets in your research, please cite:

``` 
``` 

## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.1.0
- torch-cluster == 1.4.2
- torch-geometric == 1.2.1
- torch-scatter == 1.2.0
- torch-sparse == 0.4.0
- numpy == 1.16.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.
- Kwai dataset  
```python main.py --model_name='MMGCN' --l_r=0.0005 --weight_decay=0.1 --batch_size=1024 --dim_latent=64 --num_workers=30 --aggr_mode='mean' --num_layer=2 --concat=False```
- Tiktok dataset  
`python main.py --model_name='MMGCN' --l_r=0.0005 --weight_decay=0.1 --batch_size=1024 --dim_latent=64 --num_workers=30 --aggr_mode='mean' --num_layer=2 --concat=False`
- Movielens dataset  
`python main.py --model_name='MMGCN' --l_r=0.0001 --weight_decay=0.0001 --batch_size=1024 --dim_latent=64 --num_workers=30 --aggr_mode='mean' --num_layer=2 --concat=False`  

## Dataset
We provide three processed datasets: Kwai, Tiktok, and Movielnes.  
- You can find the full version of recommendation datasets via [Kwai](https://www.kuaishou.com/activity/uimc), [Tiktok](http://ai-lab-challenge.bytedance.com/tce/vc/), and [Movielens](https://grouplens.org/datasets/movielens/).
Since the copyright of datasets, we cannot release them directly. 
To facilate the line of research, we provide some toy datasets[[BaiduPan](https://pan.baidu.com/s/1BODXP7iihw8qtxpLeEv_XA)](code: zsye) or [[GoogleDriven]](https://drive.google.com/file/d/1NoisyVDFWykTszSIbHdeoBrKn0t-D0ps/view?usp=sharing). 
Anyone needs the full datasets, please contact the owner of datasets. 



Copyright (C) <year>  Communication University of China

This program is licensed under the GNU General Public License 3.0 (https://www.gnu.org/licenses/gpl-3.0.html). Any derivative work obtained under this license must be licensed under the GNU General Public License as published by the Free Software Foundation, either Version 3 of the License, or (at your option) any later version, if this derivative work is distributed to a third party.

The copyright for the program is owned by Communication University of China. For commercial projects that require the ability to distribute the code of this program as part of a program that cannot be distributed under the GNU General Public License, please contact <taozhulin@gmail.com> to purchase a commercial license.
