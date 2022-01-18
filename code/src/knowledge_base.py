#Required libraries. 
#install deap if not installed 

#pip install deap
import itertools
from scipy.spatial import distance
import numpy as np
import math
from util import banfeld_raferty, i_index, c_index
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, Birch
import warnings
from sklearn.metrics import normalized_mutual_info_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering

from metafeatures import Meta
from deap import base
from deap import creator, tools
from scipy.stats import spearmanr
from cvi import Validation
from sklearn.preprocessing import OrdinalEncoder
import glob
from scipy.io import arff
import pandas as pd
from cvi import Validation
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from itertools import combinations

from sklearn.neighbors import NearestNeighbors




class knowledge_base:

    """
    knowledge_base is a class to create meta learning space using the datasets available in the following repo:
    https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial
    The class has three input variables, among which three are mandatory. 
    It has four methods, __init__, clus_hyp, knowledge_base, pipeline. Instantiation of the the knowledge_base

    ...

    Attributes
    ----------
    clustering_algo : list
        A list of clustering algorithm
    param_range : list
        A list of n_cluster. i.e., [2,30], [lower_limit, higher_limit]
    path_to_dataset : str
        Path to the folder that contains the datasets which would be used to create the meta learning space. Default value is None, so that user can create an instance and pass previously created
        metadataset later in the pipeline function. But for calling knowledge_base() method, this variable has to be passed.
    
    Methods
    -------
    __init__(self, clustering_algo, param_range, path_to_dataset=None)
        Initializes the variables

    clus_hyp(self)
        Sets up the configuration based on which the knowledge base would be generated. 

    knowledge_base(self)
        Method to create the the knowledge_base. The path_to_dataset variable has to be passed in order to create the knowledge_base. The function reads the file in availabe .arff format and 
        creates the knowledge base. 

    pipeline(self,test_data, knowledge_base=None)
        This method suggest the pipeline. If the knowledge_base variable is None, it will call the knowledge_base() method to generate the csv file. 

        


    """

    def __init__(self, clustering_algo, param_range, path_to_dataset=None):
        """
        Attributes
        ----------
        clustering_algo : list
            A list of clustering algorithm
        param_range : list
            A list of n_cluster. i.e., [2,30], [lower_limit, higher_limit]
        path_to_dataset : str
            Path to the folder that contains the datasets which would be used to create the meta learning space. Default value is None, so that user can create an instance and pass previously created
            metadataset later in the pipeline function. But for calling knowledge_base() method, this variable has to be passed.

        Returns
        ----------
        None
        """
        self.clustering_algo = clustering_algo
        self.param_range = param_range
        self.path_to_dataset=path_to_dataset
    
    def clus_hyp(self):
        """
        Attributes
        ----------
        Inherits attributes from init method.

        Returns
        ----------
        list
            The list clustering algorithm with the given hyperparameter range
        """
        #keeping the configuration in a list
        self.configs = list()
        for i in self.clustering_algo:

        ##K-Means/Agglo change the function each time for different meta feature creation
            #for i in list(range(100, 101,1)):
                #configs.append(i(n_clusters=i))

            for j in list(range(self.param_range[0], self.param_range[1],1)):
                self.configs.append(i(n_clusters=j)) #user can use other clustering algorithm for each phase here
        return self.configs


    def knowledge_base(self):
        """
        Attributes
        ----------
        Inherits attributes from init method.

        Returns
        ----------
        dataframe
            The dataframe of the knowledge base
        """

        configs=self.clus_hyp()

    #initializing the preprocessing methods
        self.kbin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        self.norm_l1=Normalizer(norm='l1')
        self.norm_l2=Normalizer(norm='l2')
        self.std_zscore=StandardScaler()
        self.scaling_minmax=MinMaxScaler()





        transformations = [self.kbin, self.norm_l1, self.norm_l2, self.scaling_minmax, self.std_zscore]



        df_meta=pd.DataFrame(columns=['name', 'instances', 'attributtes', 'classes', 'cat_att', 'cont_att', 'mean', 'sd', 'var', 'kurtosis', 'skewness', 'MD6', 'MD7', 'MD8', 'MD9',
           'MD10', 'MD11', 'MD12', 'MD13', 'MD14', 'MD15', 'MD16', 'MD17', 'MD18',
           'MD19', 'dataset', 'best_cluster_setting', 'transformation','multi_cvi', 'multi_cvi_correlation_score'])

    #Path to the datasets
        files=glob.glob(self.path_to_dataset+'*.arff')
        print("Number of files based on which the knowldge base would be created: ", len(files))

        for t in transformations:



            for file in files:
                print("Knowledge base is being created for the following file: ", file.split('/')[-1].split('.')[0])

                #read the dataset
                data = arff.loadarff(file)
                df = pd.DataFrame(data[0])
                df.columns = [x.lower() for x in df.columns]
                cols=df.columns
                num_cols = df._get_numeric_data().columns
                df=df.dropna() 

                #check if the dataset has column class & some preprocessing
                if df.shape[1]>2:
                    df['class'] = df['class'].str.decode('utf-8') 
                    df['class']=df['class'].replace('noise','3',regex=True)

                    try:# int(df['class'][0]): #for classes where class labels are integer
                        df['class']=df['class'].astype(str).astype(int)

                        #getting the file name
                        new_file = file.split('/')[-1].split('.')[0]
                        #print(new_file)

                    except: #for classes where the class labels are not integer
                        df_cols=df.columns
                        encoder = OrdinalEncoder()
                        # transform data
                        #result = encoder.fit_transform(df)
                        df=pd.DataFrame(result, columns=df_cols)

                        new_file = file.split('/')[-1].split('.')[0]
                        print(new_file)
        #"""
        #    Using the file name to extract metafeature, for extracting meta features, I used the csmartml package
        #    and the package requires the files to be read in CSV. So I converted the files in CSV beforehand so that we 
        #    can perform this step
        #"""
                df1=Meta("csmartml/datasets/"+new_file+".csv").extract_metafeatures(meta_type="distance")


                meta_dict=df1.to_dict('dict')   
        #"""
        #    This is the part where we apply mean and median manually as the package give some kind of erros. 
        #    Since the dataset does not have empty/ null/ NA values, so far I guess, I manually, randomly remove 20%
        #    of the data and then impute those with mean and median.     
        #"""

                #for col in df.columns:
                #    df.loc[df.sample(frac=0.2).index, col] = np.nan 

                #df.apply(lambda x: x.fillna(x.mean(), inplace=True),axis=0)

                X=df[df.columns.difference(['class'])].values
                y=df['class'].values



                X=t.fit_transform(df[df.columns.difference(['class'])].values)

        #"""
        #    The following section directly comes from SmartML homework, thatI received from Hudson. This is where
        #    the multi CVI are ranked using NSGA-2 and the corraltion scores are calculated. I tried to use/ reproduce
        #    it dunamically, but that seemed to be too much of work hence, I refrained. 

        #    The only change I made is added one more CVI to make it a combination of 3 and added the optimization 
        #    objective in the objecitve function to define the objective of the new CVI. I believe this part won't be 
        #    needed for the cSmartML project if you want to integrate preprocessing part in the tool later. 
        #"""
                def fitness_function(individual):
                    clustering = individual[0].fit(X)

                    labels = clustering.labels_
                    return Validation(X, df.values,labels).s_dbw(), davies_bouldin_score(X, labels),banfeld_raferty(X, labels)

                def initIndividual(icls, content):
                    return icls([content])

                def initPopulation(pcls, ind_init, poplist):
                    return pcls(ind_init(c) for c in poplist)

                # Think of creator as an abstraction for classes that can be modified for evolutionary algorithms.
                # Eg. in this case is the base Fitness class
                # weights = tuple defining objective of CVI, max: 1 or min: -1
                # weight should be fixed according to the objective function of the CVI, respectively
                creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0)) 
                creator.create("Individual", list, fitness=creator.FitnessMulti)

                # Toolbox contains specific functions for evolutionary operations
                # Read more: https://deap.readthedocs.io/en/master/api/base.html?highlight=toolbox#toolbox
                toolbox = base.Toolbox()
                toolbox.register("individual", initIndividual, creator.Individual)
                toolbox.register("population", initPopulation, list, toolbox.individual, configs)
                toolbox.register("evaluate", fitness_function)
                toolbox.register("select", tools.selNSGA2)

        #        """
        #            Rank configuration by multiple CVIs
        #        """
                pop = toolbox.population()

                pop_dict = {} # Numbering configurations so it's easy to compare their ranks
                for i, j in enumerate(pop):
                    pop_dict[i] = j

                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                #Fix the value for toolbox.select(pop, N), the N should be the numbe rof configurations we have from 
                #the previous cell, could have been dynamic, but I missed :) 

                nsga_sorted = toolbox.select(pop,len(configs))
                multi_cvi_rank = [list(pop_dict.values()).index(i) for i in nsga_sorted]
                multi_cvi_rank


                #CVI part



                cvi_1_dict = {}
                cvi_2_dict = {}
                cvi_3_dict = {}

                for k, v in pop_dict.items():
                    clustering = v[0].fit(X)

                    cvi_1_dict[k] = Validation(X, X, clustering.labels_).s_dbw()
                    cvi_2_dict[k] = davies_bouldin_score(X, clustering.labels_)
                    cvi_3_dict[k] = banfeld_raferty(X, clustering.labels_)

                cvi_1_dict = dict(sorted(cvi_1_dict.items(), key=lambda item: item[1], reverse=False))
                cvi_2_dict = dict(sorted(cvi_2_dict.items(), key=lambda item: item[1], reverse=False))
                cvi_3_dict = dict(sorted(cvi_3_dict.items(), key=lambda item: item[1], reverse=False))

                cvi_1_rank = list(cvi_1_dict.keys())
                cvi_2_rank = list(cvi_2_dict.keys())
                cvi_3_rank = list(cvi_3_dict.keys())


                # Rank configurations by external CVI (NMI) 

                ground_truth_dict = {}
                for k, v in pop_dict.items():
                    clustering = v[0].fit(X)
                    ground_truth_dict[k] = normalized_mutual_info_score(y, clustering.labels_)

                ground_truth_dict = dict(sorted(ground_truth_dict.items(), key=lambda item: item[1], reverse=True))
                gt_cvi_rank = list(ground_truth_dict.keys())

                cvi_1_correlation = spearmanr(cvi_1_rank, gt_cvi_rank)
                cvi_2_correlation = spearmanr(cvi_2_rank, gt_cvi_rank)
                cvi_3_correlation = spearmanr(cvi_3_rank, gt_cvi_rank)
                multi_correlation = spearmanr(multi_cvi_rank, gt_cvi_rank)



                #Finding out the categorical columns
                cat_cols=df.iloc[:, :-1].columns
                cat_num_cols = df.iloc[:, :-1]._get_numeric_data().columns



                #pushing everythign to the dataframe to create the knowledge base
                df_meta=df_meta.append({'name': new_file, 
                                     'instances':df.shape[0], 
                                     'attributtes':df.shape[1]-1, 
                                     'classes':df['class'].nunique(), 

                                     'cat_att':len(list(set(cat_cols) - set(cat_num_cols))),
                                     'cont_att':len(num_cols),

                                    'mean': meta_dict.get('mean')[0],
                                    'sd': meta_dict.get('sd')[0],
                                    'var': meta_dict.get('var')[0],
                                    'kurtosis': meta_dict.get('kurtosis')[0],
                                    'skewness': meta_dict.get('skewness')[0],
                                    'MD6': meta_dict.get('MD6')[0],
                                    'MD7': meta_dict.get('MD7')[0],
                                        'MD8': meta_dict.get('MD8')[0],
                                        'MD9': meta_dict.get('MD9')[0],
                                        'MD10': meta_dict.get('MD10')[0],
                                        'MD11': meta_dict.get('MD11')[0],
                                        'MD12': meta_dict.get('MD12')[0],
                                        'MD13': meta_dict.get('MD13')[0],
                                        'MD14': meta_dict.get('MD14')[0],
                                        'MD15': meta_dict.get('MD15')[0],
                                        'MD16': meta_dict.get('MD16')[0],
                                        'MD17': meta_dict.get('MD17')[0],
                                        'MD18': meta_dict.get('MD18')[0],
                                        'MD19': meta_dict.get('MD19')[0],
                                        'best_cluster_setting':pop_dict[multi_cvi_rank[0]],
                                        'transformation':str("mean + "+str(t)),
                                        'multi_cvi':'banfeld_raferty, davies_bouldin_score, SDbw',


                                        'multi_cvi_correlation_score': multi_correlation.correlation}, ignore_index=True)

        df_meta.to_csv('meta_data_new.csv')
        return df_meta


    def pipeline(self,test_data, knowledge_base=None):
        """
        Attributes
        ----------
        test_data: str
            A string of the path to test dataset. The dataset has to be in CSV format. 
        knowledge_base: str
            A string of the path to the knowledge_base metadataset. The dataset has to be in CSV format. The previous knowledge_base() method creates the metadataset CSV file which later could be passed
            here. 

        Returns
        ----------
        str, str
            String of the suggested cluster setting
            String of the suggested data preprocessing pipeline
        """
        
        if knowledge_base==None:
            new_df=self.knowledge_base()
        else:
            new_df=pd.read_csv(knowledge_base, na_values='?')
        
        new_df_2=new_df[['mean', 'sd', 'var', 'kurtosis',
       'skewness', 'MD6', 'MD16', 'MD17', 'MD18']]

        data = pd.read_csv(test_data, na_values='?')

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_meta_org = Meta(test_data).extract_metafeatures(meta_type="distance")
        X_meta=X_meta_org[['mean', 'sd', 'var', 'kurtosis',
       'skewness', 'MD6', 'MD16', 'MD17', 'MD18']]
        
        samples = new_df_2.iloc[:,:-1].to_numpy()
        query = X_meta.iloc[:,:-1].to_numpy()

        neigh = NearestNeighbors(n_neighbors=505, metric='euclidean')
        neigh.fit(samples)

        nn = neigh.kneighbors(query, 51, return_distance=False)

        datasets = new_df[new_df.index.isin(nn.tolist()[0])]["name"].values.tolist()

        matched=new_df[new_df['name']==datasets[0]].sort_values(by=['multi_cvi_correlation_score'], ascending=False).iloc[0]


        
        return matched['best_cluster_setting'], matched['transformation']