Code for the paper - [Multi-Variate Time Series Forecasting on Variable Subsets]() accepted at KDD 2022 Research Track.

# Running the model

Datasets - METR-LA, SOLAR, TRAFFIC, ECG. This code provides a running example with all components on [MTGNN](https://github.com/nnzhan/MTGNN) model (we acknowledge the authors of the work).

## Standard Training
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 {3} --mask_remaining {4}
```
Here, <br />
{0} - refers to the dataset directory: ./data/{ECG/TRAFFIC/METR-LA/SOLAR} <br />
{1} - refers to the model name <br />
{2} - refers to the manually assigned "ID" of the experiment  <br />
{3} - step_size1 is 2500 for METR-LA and SOLAR, 400 for ECG, 1000 for TRAFFIC <br />
{4} - inference post training in the partial setting, set to true or false. Note - mask_remaining is the alias for "Partial" setting in the paper
* random_node_idx_split_runs - the number of randomly sampled subsets per trained model run
* lower_limit_random_node_selections and upper_limit_random_node_selections - the percentage of variables in the subset **S**.


### Training with predefined subset S, the S apriori setting
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 50 --predefined_S --random_node_idx_split_runs 1 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```


### Training the model with Identity matrix as Adjacency
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --adj_identity_train_test --random_node_idx_split_runs 100 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```


## Inference

### Partial setting inference
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --mask_remaining True
```
* Note that epochs are set to 0 and mask_remaining (alias of "Partial" setting in the paper) to True


### Oracle setting inference
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --do_full_set_oracle true --full_set_oracle_lower_limit 15 --full_set_oracle_upper_limit 15
```


## Our Wrapper Technique
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --borrow_from_train_data true --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True
```


## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt


## Data Preparation


### Multivariate time series datasets

Download Solar and Traffic datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

Download the METR-LA dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git). Move them into the data folder. (Optinally - download the adjacency matrix for META-LA from [here](https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl) and put it as ./data/sensor_graph/adj_mx.pkl , as shown below):
```
wget https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl
mkdir data/sensor_graph
mv adj_mx.pkl data/sensor_graph/
```

Download the ECG5000 dataset from [time series classification](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

```

# Create data directories
mkdir -p data/{METR-LA,SOLAR,TRAFFIC,ECG}

# for any dataset, run the following command
python generate_training_data.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```
Here <br />
{0} is for the dataset: metr-la, solar, traffic, ECG <br />
{1} is the directory where to save the train, valid, test splits. These are created from the first command <br />
{2} the raw data filename (the downloaded file), such as - ECG_data.csv, metr-la.hd5, solar.txt, traffic.txt


## Citation

```
@inproceedings{10.1145/3534678.3539394,
author = {Chauhan, Jatin and Raghuveer, Aravindan and Saket, Rishi and Nandy, Jay and Ravindran, Balaraman},
title = {Multi-Variate Time Series Forecasting on Variable Subsets},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539394},
doi = {10.1145/3534678.3539394},
abstract = {We formulate a new inference task in the domain of multivariate time series forecasting (MTSF), called Variable Subset Forecast (VSF), where only a small subset of the variables is available during inference. Variables are absent during inference because of long-term data loss (eg. sensor failures) or high -> low-resource domain shift between train / test. To the best of our knowledge, robustness of MTSF models in presence of such failures, has not been studied in the literature. Through extensive evaluation, we first show that the performance of state of the art methods degrade significantly in the VSF setting. We propose a non-parametric, wrapper technique that can be applied on top any existing forecast models. Through systematic experiments across 4 datasets and 5 forecast models, we show that our technique is able to recover close to 95% performance of the models even when only 15% of the original variables are present.},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {76–86},
numpages = {11},
keywords = {multivariate time series forecasting, variable subsets, partial inference, retrieval model},
location = {Washington DC, USA},
series = {KDD '22}
}
```
