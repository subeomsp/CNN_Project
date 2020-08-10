# CNN_Project

CNN for Sentence Classification (Naver Movie sentiment analysis)
* language : python

**How to use**<br>
**CNN modeling with mecab word2vec model**
1. clone repository to your local computer
2. recommend to convert file extension to .ipynb
3. recommend to execute in jupyter notebook or google colaboratory environment
4. you can use raw ratings.txt or ratings_test.txt and ratings_test.txt in 'prep for modeling' folder
5. make word2vec model first with tokenize_mecab_300dim_w_sw.py file
6. make cnn model with word2vec model
7. you can use 'stopwords.xlsx' in 'prep for modeling' folder for removing stop words (in my case, there was no big difference in val_accuracy)
