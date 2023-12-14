import os
import pandas as pd
import numpy as np

class FunkSVDCF:
    def __init__(self, num_users, num_items, num_factors, learning_rate, regularization, num_epochs):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.alpha = learning_rate
        self.lambda_ = regularization
        self.num_epochs = num_epochs
        
        # Initialize model parameters
        self.global_bias = 0
        self.user_biases = np.zeros(num_users)
        self.item_biases = np.zeros(num_items)
        self.user_factors = np.random.normal(scale=0.1, size=(num_users, num_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(num_items, num_factors))

    def fit(self, df):
        # Convert the DataFrame of implicit interactions into a list of tuples
        interactions = df[['user_id', 'movie_id']].values.tolist()

        unique_user_ids = df.user_id.unique()
        user_id_dict = {user_id: index for index, user_id in enumerate(unique_user_ids)}
        self.user_id_index = user_id_dict

        unique_movie_ids = df.movie_id.unique()
        movie_id_dict = {movie_id: index for index, movie_id in enumerate(unique_movie_ids)}
        self.movie_id_index = movie_id_dict

        for epoch in range(self.num_epochs):
            np.random.shuffle(interactions)
            
            for user, item in interactions:
                user_index = self.user_id_index[user]
                item_index = self.movie_id_index[item]

                user_bias = self.user_biases[user_index]
                item_bias = self.item_biases[item_index]
                user_factor = self.user_factors[user_index, :]
                
                # Since it's implicit feedback, we consider the interaction as a positive signal (rating = 1)
                predicted_interaction = self.global_bias + user_bias + item_bias + np.dot(user_factor, self.item_factors[item_index, :])
                error = 1 - predicted_interaction  # Implicit interaction is considered as 1
                
                # Update biases and factors
                self.global_bias += self.alpha * (error - self.lambda_ * self.global_bias)
                self.user_biases[user_index] += self.alpha * (error - self.lambda_ * user_bias)
                self.item_biases[item_index] += self.alpha * (error - self.lambda_ * item_bias)
                self.user_factors[user_index, :] += self.alpha * (error * self.item_factors[item_index, :] - self.lambda_ * user_factor)
                self.item_factors[item_index, :] += self.alpha * (error * user_factor - self.lambda_ * self.item_factors[item_index, :])
    
    def predict(self, user):
        user_index = self.user_id_index[user]
        item_index = self.movie_id_index[item]

        prediction = self.global_bias + self.user_biases[user_index] + self.item_biases[item_index] + np.dot(self.user_factors[user_index, :], self.item_factors[item_index, :])

        return prediction
    
    def add_new_user(self, new_user_id, interactions):
        # Initialize the biases and factors for the new user
        self.user_biases = np.append(self.user_biases, 0)  # Initialize user bias as 0
        self.user_factors = np.vstack([self.user_factors, np.random.normal(scale=0.1, size=(1, self.num_factors))])  # Initialize user factors with small random values
        
        if new_user_id not in self.user_id_index.keys():
            self.user_id_index[new_user_id] = len(self.user_id_index)

        user_index = self.user_id_index[new_user_id]

        # Update the model parameters for the new user based on the new interactions
        for item in interactions:
            item_index = self.movie_id_index[item]

            user_bias = self.user_biases[user_index]
            item_bias = self.item_biases[item_index]
            user_factor = self.user_factors[user_index, :]
            
            predicted_interaction = self.global_bias + user_bias + item_bias + np.dot(user_factor, self.item_factors[item_index, :])
            error = 1 - predicted_interaction  # Implicit interaction is considered as 1
            
            # Update biases and factors for the new user
            self.global_bias += self.alpha * (error - self.lambda_ * self.global_bias)
            self.user_biases[user_index] += self.alpha * (error - self.lambda_ * user_bias)
            self.item_biases[item_index] += self.alpha * (error - self.lambda_ * item_bias)
            self.user_factors[user_index, :] += self.alpha * (error * self.item_factors[item_index, :] - self.lambda_ * user_factor)
    
    def recommend_items(self, user_id, interactions, num_recommendations=10):
        user_index = self.user_id_index[user_id]

        # Calculate the predicted interaction scores for all items for the given user
        scores = self.global_bias + self.user_biases[user_index] + self.item_biases + np.dot(self.item_factors, self.user_factors[user_index, :])
        
        # Get the indices of the top scored items
        movie_indices = [self.movie_id_index[watched_movie] for watched_movie in interactions]
        rec_movie_indices = [movie_index for movie_index in scores.argsort()[::-1] if movie_index not in movie_indices][:num_recommendations]

        return rec_movie_indices
    

