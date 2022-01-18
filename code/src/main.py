
from knowledge_base import knowledge_base
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def main():
    """
    Instance creation requires three input 
    #input: list
                Mandatory input of list of clustering algorithm, 
            list
                Mandatory input of list of lower and higher limit of number of cluster (n_cluster), 
            str
                Default None, provided user already has required knowledge base. 
                Or string to path_to_datasets based on which the knowledge base would be created if user wants to create the knowledge base. 

    #for datasets I put all the files found here: I used https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial repo
    #in a local repo
    """


    #Instance creation of knowledge_base class. The following example only suggests the clustering algorithm and data preprocessing pipeline for arrhythmia.csv depending on the 
    #already available knowledgebase, meta_data.csv file without creating the file.
    p1 = knowledge_base([KMeans, AgglomerativeClustering], [2,30], None) 


    

    cluster_setting, transformations=p1.pipeline( 'sample_test_data/arrhythmia.csv', 'meta_data.csv')
    print('Suggested cluster setting:', cluster_setting)
    print('Suggested data preprocessing pipeline:', transformations)

    # if user wants to create the knowledge base, uncomment the following two lines and comment the previous three lines. 
    #p1= knowledge_base([KMeans, AgglomerativeClustering], [2,30], 'path_to_datasets_where the .arff file from the git repo resides/') 
    #cluster_setting, transformations=p1.pipeline( 'sample_test_data/arrhythmia.csv')



if __name__ == "__main__":
    main()

        


