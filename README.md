# introduction of files 

## project--article classification:

   aim: 
   
      use text classification to predict the most relevant news articles for each of the 10 topics. 
    
   description:
    
    As Data Scientist, you are tasked to help these users find the most interesting articles according to their preferred topics. 
    You have a training dataset containing about 9500 news articles, each assigned to one of the above topics. 
    In addition, (as in real life situation) the dataset contains about 48% of irrelevant articles (marked as IRRELEVANT) that do not belong to any of the topics;       
    hence the users are not interested in them. The distribution of articles over topics is not uniform. 
    There are some topics with large number of articles, and some with very small number.

    One day, 500 new articles have been published. This is your test set that has similar article distribution over topics to the training set. 
    Your task is to suggest up to 10 of the most relevant articles from this set of 500 to each user. 
    The number of suggestions is limited to 10, because, presumably, the users do not want to read more suggestions. 
    It is possible, however, that some topics within this test set have less than 10 articles. 
    You also do not want to suggest 10 articles if they are unlikely to be relevant, because you are concerned that the users may get discouraged and stop using your application altogether. 
    Therefore you need to take a balanced approach, paying attention to not suggesting too many articles for rare topics.


## project-- product quantization algorithm implementation with L1 distance:

only support the following modules/libraries:

      Scipy 1.4.1
      Numpy 1.81.6
      Python 3.6

### part1: PQ for L1 Distance

This section will implement the product quantization method with L1 distance as the distance function. 

      **Note** that due to the change of distance function, the PQ method introduced in the class no longer works. You need to work out how to adjust the method and make it work for $L_1$ distance. For example, the K-means clustering algorithm works for $L_2$ distance, you need to implement its $L_1$ variants (we denote it as K-means* in this project). You will also need to explain your adjustments in the report later.

Specifically, you are required to write a method `pq()` in the file `submission.py` that takes FOUR arguments as input:

      1. **data** is an array with shape (N,M) and dtype='float32', where N is the number of vectors and M is the dimensionality.
      2. **P** is the number of partitions/blocks the vector will be split into. Note that in the examples from the inverted multi index paper, P is set to 2. But in this project, you are required to implement a more general case where P can be any integer >= 2. You can assume that P is always divides M in this project. 
      3. **init_centroids** is an array with shape (P,K,M/P) and dtype='float32', which corresponds to the initial centroids for P blocks. For each block, K M/P-dim vectors are used as the initial centroids.
      **Note** that in this project, K is fixed to be 256.
      4. **max_iter** is the maximum number of iterations of the K-means* clustering algorithm. 
      **Note** that in this project, the stopping condition of K-means* clustering is that the algorithm has run for ```max_iter``` iterations.

return format:

      The `pq()` method returns a codebook and codes for the data vectors, where
      * **codebooks** is an array with shape (P, K, M/P) and dtype='float32', which corresponds to the PQ codebooks for the inverted multi-index. E.g., there are P codebooks and each one has K M/P-dimensional codewords.
      * **codes** is an array with shape (N, P) and dtype=='uint8', which corresponds to the codes for the data vectors. The dtype='uint8' is because K is fixed to be 256 thus the codes should integers between 0 and 255. 
  
  
### part2: Query using Inverted Multi-index with L1 Distance

This section will implement the query method using the idea of inverted multi-index with L1 distance. 
Specifically, you are required to write a method `query()` in the file `submission.py` that takes arguments as input:

      1. **queries** is an array with shape (Q, M) and dtype='float32', where Q is the number of query vectors and M is the dimensionality.
      2. **codebooks** is an array with shape (P, K, M/P) and dtype='float32', which corresponds to the `codebooks` returned by `pq()` in part 1.
      3. **codes** is an array with shape (N, P) and dtype=='uint8', which corresponds to the `codes` returned by `pq()` in part 1.
      4. **T** is an integer which indicates the minimum number of returned candidates for each query. 

return format:

      The `query()` method returns an array contains the candidates for each query. Specifically, it returns
      **candidates** is a list with Q elements, where the i-th element is a **set** that contains at least T integers, corresponds to the id of the candidates of the i-th query. 
      For example, assume $T=10$, for some query we have already obtained $9$ candidate points. Since $9 < T$, the algorithm continues. Assume the next retrieved cell contains $3$ points, then the returned set will contain $12$ points in total.

