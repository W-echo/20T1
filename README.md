# project--article classification:

   aim: 
   
      use text classification to predict the most relevant news articles for each of the 10 topics. 
    
   description:
    
    As Data Scientist, you are tasked to help these users find the most interesting articles according to their preferred topics. 
    You have a training dataset containing about 9500 news articles, each assigned to one of the above topics. 
    In addition, (as in real life situation) the dataset contains about 48% of irrelevant articles (marked as IRRELEVANT) that do not belong to any of the topics;       hence the users are not interested in them. The distribution of articles over topics is not uniform. 
    There are some topics with large number of articles, and some with very small number.

    One day, 500 new articles have been published. This is your test set that has similar article distribution over topics to the training set. 
    Your task is to suggest up to 10 of the most relevant articles from this set of 500 to each user. 
    The number of suggestions is limited to 10, because, presumably, the users do not want to read more suggestions. 
    It is possible, however, that some topics within this test set have less than 10 articles. 
    You also do not want to suggest 10 articles if they are unlikely to be relevant, because you are concerned that the users may get discouraged and stop using your application altogether. 
    Therefore you need to take a balanced approach, paying attention to not suggesting too many articles for rare topics.
