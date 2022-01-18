# Code

This repository contains codes for the thesis project titled, **Meta Learning for Automated Data Pre-processing for clustering**. The source code is in the knowledge_base class residing in the ```knowledge_base.py``` file written in **python3**, specifically ```Python 3.9.7```.

>This guide assumes user already has ```Python 3.9.7``` installed in the system. If user has both ```Python 2*``` and ```Python 3*```installed in the system, user should specify which ```pip``` or ```python``` command to run. Just ```pip``` will run **pip2** and just ```python``` would run **python2**. ```pip3``` and ```python3``` will run **pip3** and **python3** respectively. 

```knowledge_base.py``` is a Python library that creates a meta learning space based on datasets found in [this repository](https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial). The file is read in available **.arff** format. It also has a method named ```pipeline()``` which suggest the clustering algorithm and data preprocessing for a given test dataset. Details of the class, methods, and variables can be found in the file itself. User can skip the time consuming knowledge base creation and proceed with the available knowledge base, ```meta_data.csv``` already created. 

## Dependencies
The knowledge_base.py package depends on various python packages;
1. ```deap==1.3.1```
2. ```numpy==1.19.0```
3. ```pandas==1.2.4```
4. ```s_dbw==0.4.0```
5. ```scikit_learn==1.0.2```
6. ```scipy==1.7.0```


It depends on the file;
1. ```util.py```

It also depends on the cSmartML libraries; 

1) ```cvi.py```
2) ```metafeatures.py```
3) ```sdbw.py```

All these files have to be in the same folder in order to successfully run the ```main.py```. The files are arranged in an order so user can clone the whole repository and follow the following steps to create the knowledge base and/ or receive the suggested pipeline.

 


## Run the Code
In order to run the example code, user need the datasets and dataset repositories included in the ```../src/``` repo. 

### 1. Clone the repository


```bash
git clone https://github.com/mbahmani/csmartml
```

After cloning, go to the the respected folder in your cloned local repository. 

```bash
cd path_to_csmartml/code/src
```

### 2. Install the dependencies
Then make sure to install all the dependencies. The following cell will install all the required libraries in the current environment (usually root if not already in a virtual environment) user is using. It is suggested to create virtual environment and install the libraries there to avoid further conflicts created by any other package that depends on these. A short guide to virtual environment creation using [Conda](https://docs.conda.io/en/latest/) can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). And then user has to activate the virtual environment and install the libraries there. 

```bash
pip install -r requirements.txt 
```

### 3. Run main.py
Now after the dependencies are installed and dependant datasets are accordingly set (user should not do anything about the datasets). Running the following cell should suggest the user with data preprocessing pipeline and clustering algorithm. 

```bash
python main.py 
```


## Usage

```python
from knowledge_base import knowledge_base

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


# create an instance of the knowledge_base class if user want to use previously created meta learning dataset
# [2, 30] 2 is the lower limit of cluster number and 30 is the higher limit of cluster number
p1 = knowledge_base([KMeans, AgglomerativeClustering], [2,30], None)

# create an instance of the knowledge_base if user want to create the meta learning dataset
p1 = knowledge_base([KMeans, AgglomerativeClustering], [2,30],'path_to_datasets')

# create the meta learning space. It will save a new meta_data_new.csv file in the directory. 
# also returns the dataframe
p1.knowledge_base() 

# suggests the pipeline for test dataset, arrhythmia.csv if path of meta dataset is passed.
# returns the cluster_setting, preprocessing_pipeline
p1.pipeline( 'sample_test_data/arrhythmia.csv', 'meta_data.csv')

# if meta dataset path is not given,
# it will call the knowledge_base() method within itself and create the knowledge base and 
# use the file to make the suggestion.
p1.pipeline( 'sample_test_data/arrhythmia.csv')
```
## Contributing
Pull requests are welcome. 

User can play with the clustering algorithm, but make sure that the algorithm has ```n_cluster``` attribute. Or change the variable accordingly in case of more hyper parameter if required. 
Please make sure to update dependant methods as required.

## License
[MIT](https://choosealicense.com/licenses/mit/)