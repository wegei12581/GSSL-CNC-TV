# GSSL-CNC-TV
In our paper, we introduce a semi-supervised learning model based on CNC-TV based on GraphBGS.
The associated GraphBGS can be seen in <https://github.com/jhonygiraldo/GraphBGS>, the corresponding article is <https://ieeexplore.ieee.org/document/9412999/>.

## Small data experiment
In our code, we can compare the results of TV, Tik and CNC models on small data sets by running demo_gssl_cnc_tv.

## Numerical Experiments
In order to validate our model 's ability to reproduce real signals, we conduct experiments on CDNet2014, a dataset encompassing a vast array of data from various aspects and we can download the corresponding data set through <http://jacarini.dinf.usherbrooke.ca/dataset2014/>, then we  compare the experimental outcomes with those of alternative models.


In the experiment, we primarily proceed through the following steps to complete the experiment and compute the corresponding performance indicators.
1).instance segmentation, 2).background initialization, 3).nodes representation, 4).graph construction, 5).graph signal sampling, 6)graph signal recovery(using our semi-supervised based on CNC-TV model), 7).Computer performance index. 

