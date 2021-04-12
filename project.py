
#importing the required libraries
import numpy as np
import pandas as pd
import pickle
#import matrix_factorization_utilities
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from flask import Flask, render_template, request
from IPython.display import HTML

def best_movies_by_genre(genre,top_n):
    movie_score = pd.read_csv('movie_score.csv')
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/genre", methods = ['GET','POST'])
def genre():
    if request.method == 'POST':
      result = request.form
      print(result['Genre'])
      print(type(result['Genre']))
      df = best_movies_by_genre(result['Genre'],10)
      df.reset_index(inplace=True)
      df = df.drop(labels='index', axis=1)
      html = HTML(df.to_html(classes='table table-striped'))
      dummy = {}
      dummy[0] = html
    #   return str(html)
      return render_template("genre.html",result = dummy, gename = {1:result['Genre']})
    
if __name__ == "__main__":
    app.run(debug=True)

def init():
    movie_score = pd.read_csv('movie_score.csv')
    ratings_movies = pd.read_csv('ratings_movies.csv')
    movie_content_df_temp = pd.read_csv('mv_cnt_tmp.csv')
    a_file = open("indicies.pkl", "rb")
    inds = pickle.load(a_file)
    a_file.close()
    print(inds['Skyfall (2012)'])
    rev_ind = {}
    for key,val in inds.items():
        rev_ind[val] = key
    from numpy import load
    data_dict = load('cosine.npz')
    cosine_sim = data_dict['arr_0']
    #ratings_movies.head()

#movie_score.head()

# Gives the best movies according to genre based on weighted score which is calculated using IMDB formula


# best_movies_by_genre('Musical',10)  

# Gets the other top 10 movies which are watched by the people who saw this particular movie

def get_other_movies(movie_name):
    ratings_movies = pd.read_csv('ratings_movies.csv')
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    return other_users_watched[1:11]

# get_other_movies('Gone Girl (2014)')



# Directly getting top 10 movies based on content similarity
# cosine_sim

def get_similar_movies_based_on_content(movie_name) :
    movie_content_df_temp = pd.read_csv('mv_cnt_tmp.csv')
    a_file = open("indicies.pkl", "rb")
    inds = pickle.load(a_file)
    a_file.close()
    print(inds['Skyfall (2012)'])
    rev_ind = {}
    for key,val in inds.items():
        rev_ind[val] = key
    from numpy import load
    data_dict = load('cosine.npz')
    cosine_sim = data_dict['arr_0']
    movie_index = inds[movie_name]
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:11]
    print(sim_scores)
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    if(movie_index in movie_indices):
        movie_indices.remove(movie_index)
    print(movie_indices)
    similar_movies = pd.DataFrame(movie_content_df_temp[['title','genres']].iloc[movie_indices])
    return similar_movies[:10]

# get_similar_movies_based_on_content('Skyfall (2012)')



# #get ordered list of movieIds
# item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
# #add in data frame index value to data frame
# item_indices['movie_index']=item_indices.index
# #inspect data frame
# item_indices.head()


# # In[166]:


# #get ordered list of userIds
# user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
# #add in data frame index value to data frame
# user_indices['user_index']=user_indices.index
# #inspect data frame
# user_indices.head()


# # In[167]:


# ratings.head()


# # In[168]:


# #join the movie indices
# df_with_index = pd.merge(ratings,item_indices,on='movieId')
# #join the user indices
# df_with_index=pd.merge(df_with_index,user_indices,on='userId')
# #inspec the data frame
# df_with_index.head()


# # In[169]:


# #import train_test_split module
# from sklearn.model_selection import train_test_split
# #take 80% as the training set and 20% as the test set
# df_train, df_test= train_test_split(df_with_index,test_size=0.2)
# print(len(df_train))
# print(len(df_test))


# # In[170]:


# df_train.head()


# # In[171]:


# df_test.head()


# # In[172]:


# n_users = ratings.userId.unique().shape[0]
# n_items = ratings.movieId.unique().shape[0]
# print(n_users)
# print(n_items)


# # #### User_index is row and Movie_index is column and value is rating

# # In[176]:


# #Create two user-item matrices, one for training and another for testing
# train_data_matrix = np.zeros((n_users, n_items))
#     #for every line in the data
# for line in df_train.itertuples():
#     #set the value in the column and row to 
#     #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
#     train_data_matrix[line[5], line[4]] = line[3]
# train_data_matrix.shape


# # In[177]:


# #Create two user-item matrices, one for training and another for testing
# test_data_matrix = np.zeros((n_users, n_items))
#     #for every line in the data
# for line in df_test[:1].itertuples():
#     #set the value in the column and row to 
#     #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
#     #print(line[2])
#     test_data_matrix[line[5], line[4]] = line[3]
#     #train_data_matrix[line['movieId'], line['userId']] = line['rating']
# test_data_matrix.shape


# # In[178]:


# pd.DataFrame(train_data_matrix).head()


# # In[179]:


# df_train['rating'].max()


# # In[180]:


# from sklearn.metrics import mean_squared_error
# from math import sqrt
# def rmse(prediction, ground_truth):
#     #select prediction values that are non-zero and flatten into 1 array
#     prediction = prediction[ground_truth.nonzero()].flatten() 
#     #select test values that are non-zero and flatten into 1 array
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     #return RMSE between values
#     return sqrt(mean_squared_error(prediction, ground_truth))


# # In[181]:


# #Calculate the rmse sscore of SVD using different values of k (latent features)
# from scipy.sparse.linalg import svds

# rmse_list = []
# for i in [1,2,5,20,40,60,100,200]:
#     #apply svd to the test data
#     u,s,vt = svds(train_data_matrix,k=i)
#     #get diagonal matrix
#     s_diag_matrix=np.diag(s)
#     #predict x with dot product of u s_diag and vt
#     X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
#     #calculate rmse score of matrix factorisation predictions
#     rmse_score = rmse(X_pred,test_data_matrix)
#     rmse_list.append(rmse_score)
#     print("Matrix Factorisation with " + str(i) +" latent features has a RMSE of " + str(rmse_score))


# # In[182]:


# #Convert predictions to a DataFrame
# mf_pred = pd.DataFrame(X_pred)
# mf_pred.head()


# # In[183]:


# df_names = pd.merge(ratings,movie_list,on='movieId')
# df_names.head()


# # In[184]:


# #choose a user ID
# user_id = 1
# #get movies rated by this user id
# users_movies = df_names.loc[df_names["userId"]==user_id]
# #print how many ratings user has made 
# print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
# #list movies that have been rated
# users_movies


# # In[185]:


# user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
# #get movie ratings predicted for this user and sort by highest rating prediction
# sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
# #rename the columns
# sorted_user_predictions.columns=['ratings']
# #save the index values as movie id
# sorted_user_predictions['movieId']=sorted_user_predictions.index
# print("Top 10 predictions for User " + str(user_id))
# #display the top 10 predictions for this user
# pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]


# # In[186]:


# #count number of unique users
# numUsers = df_train.userId.unique().shape[0]
# #count number of unitque movies
# numMovies = df_train.movieId.unique().shape[0]
# print(len(df_train))
# print(numUsers) 
# print(numMovies) 


# # In[187]:


# #Separate out the values of the df_train data set into separate variables
# Users = df_train['userId'].values
# Movies = df_train['movieId'].values
# Ratings = df_train['rating'].values
# print(Users),print(len(Users))
# print(Movies),print(len(Movies))
# print(Ratings),print(len(Ratings))


# # In[194]:


# #import libraries
# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Embedding, Reshape 
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint


# # In[195]:


# from keras.utils import plot_model


# # In[196]:


# # Couting no of unique users and movies
# len(ratings.userId.unique()), len(ratings.movieId.unique())


# # In[197]:


# # Assigning a unique value to each user and movie in range 0,no_of_users and 0,no_of_movies respectively.
# ratings.userId = ratings.userId.astype('category').cat.codes.values
# ratings.movieId = ratings.movieId.astype('category').cat.codes.values


# # In[198]:


# # Splitting the data into train and test.
# train, test = train_test_split(ratings, test_size=0.2)


# # In[199]:


# train.head()


# # In[200]:


# test.head()


# # In[201]:


# n_users, n_movies = len(ratings.userId.unique()), len(ratings.movieId.unique())


# # In[204]:


# # Returns a neural network model which performs matrix factorisation
# def matrix_factorisation_model_with_n_latent_factors(n_latent_factors) :
#     movie_input = keras.layers.Input(shape=[1],name='Item')
#     movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
#     movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

#     user_input = keras.layers.Input(shape=[1],name='User')
#     user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))
#     prod = keras.layers.dot([movie_vec, user_vec], axes=1)
    
#     model = keras.Model([user_input, movie_input], prod)
#     model.compile('adam', 'mean_squared_error')
    
#     return model


# # In[205]:


# model = matrix_factorisation_model_with_n_latent_factors(20)


# # In[206]:


# model.summary()


# # In[ ]:


# #Training the model
# history = model.fit([train.userId, train.movieId], train.rating, epochs=50, verbose=0)


# # In[391]:


# y_hat = np.round(model.predict([test.userId, test.movieId]),0)
# y_true = test.rating


# # In[392]:


# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(y_true, y_hat)


# # In[393]:


# #Getting summary of movie embeddings
# movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
# pd.DataFrame(movie_embedding_learnt).describe()


# # In[394]:


# # Getting summary of user embeddings from the model
# user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
# pd.DataFrame(user_embedding_learnt).describe()


# # In[395]:


# from keras.constraints import non_neg


# # In[396]:


# # Returns a neural network model which performs matrix factorisation with additional constraint on embeddings(that they can't be negative)
# def matrix_factorisation_model_with_n_latent_factors_and_non_negative_embedding(n_latent_factors) :
#     movie_input = keras.layers.Input(shape=[1],name='Item')
#     movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Non-Negative-Movie-Embedding',embeddings_constraint=non_neg())(movie_input)
#     movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

#     user_input = keras.layers.Input(shape=[1],name='User')
#     user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='Non-Negative-User-Embedding',embeddings_constraint=non_neg())(user_input))
#     prod = keras.layers.merge([movie_vec, user_vec], mode='dot',name='DotProduct')
    
#     model = keras.Model([user_input, movie_input], prod)
#     model.compile('adam', 'mean_squared_error')
    
#     return model


# # In[397]:


# model2 = matrix_factorisation_model_with_n_latent_factors_and_non_negative_embedding(5)


# # In[412]:


# model2.summary()


# # In[398]:


# history_nonneg = model2.fit([train.userId, train.movieId], train.rating, epochs=50, verbose=0)


# # In[399]:


# movie_embedding_learnt = model2.get_layer(name='Non-Negative-Movie-Embedding').get_weights()[0]
# pd.DataFrame(movie_embedding_learnt).describe()


# # In[401]:


# y_hat = np.round(model2.predict([test.userId, test.movieId]),0)
# y_true = test.rating


# # In[402]:


# mean_absolute_error(y_true, y_hat)


# # In[409]:


# # Returns a neural network model which does recommendation
# def neural_network_model(n_latent_factors_user, n_latent_factors_movie):
    
#     movie_input = keras.layers.Input(shape=[1],name='Item')
#     movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
#     movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
#     movie_vec = keras.layers.Dropout(0.2)(movie_vec)


#     user_input = keras.layers.Input(shape=[1],name='User')
#     user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
#     user_vec = keras.layers.Dropout(0.2)(user_vec)


#     concat = keras.layers.merge([movie_vec, user_vec], mode='concat',name='Concat')
#     concat_dropout = keras.layers.Dropout(0.2)(concat)
#     dense = keras.layers.Dense(100,name='FullyConnected')(concat)
#     dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
#     dense_2 = keras.layers.Dense(50,name='FullyConnected-1')(concat)
#     dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
#     dense_3 = keras.layers.Dense(20,name='FullyConnected-2')(dense_2)
#     dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
#     dense_4 = keras.layers.Dense(10,name='FullyConnected-3', activation='relu')(dense_3)


#     result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
#     adam = Adam(lr=0.005)
#     model = keras.Model([user_input, movie_input], result)
#     model.compile(optimizer=adam,loss= 'mean_absolute_error')
#     return model

# model3 = neural_network_model(10,13)

# history_neural_network = model3.fit([train.userId, train.movieId], train.rating, epochs=50, verbose=0)

# model3.summary()

# y_hat = np.round(model3.predict([test.userId, test.movieId]),0)
# y_true = test.rating
# mean_absolute_error(y_true, y_hat)