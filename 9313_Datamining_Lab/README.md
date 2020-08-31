
# Lab2: implement and Optimize BUC algorithm 

## goal
    need to implement the full `buc_rec_optimized` algorithm with the single-tuple optimization (as described below). Given an input dataframe:

         A | B | M 
        ---|---|---
         1 | 2 | 100
         2 | 1 | 20


    Invoking  `buc_rec_optimized` on this data will result in following dataframe: 

         A | B | M
        ---|---|---
         1 | 2 | 100
         1 |ALL| 100
         2 | 1 | 20 
         2 |ALL| 20
        ALL| 1 | 20
        ALL| 2 | 100
        ALL|ALL| 120

## The single-tuple optimization

    Consider the naive way of recursive implementation of the BUC algorithm, you will notice that it uses several recursive calls to compute all the derived results from an input that consists of only one tuple. This is certainly a waste of computation. 

    For example, if we are asked to compute the cube given the following input

     B | C | M 
    ---|---|---
     1 | 2 | 100

    We can immmediately output the following, **without** using any recursive calls. 

    1    2    100
    *    2    100
    1    *    100
    *    *    100
    
    
# Lab3:  Hierarchical Clustering with **complete link**.

![image](https://github.com/W-echo/20T1/blob/master/9313_Datamining_Lab/Lab3_Hierarchical_Clustering/complete_link.png)
