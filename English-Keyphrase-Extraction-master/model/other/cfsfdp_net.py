import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to calculate distance matrix
def calculate_distance_matrix(data):
    num_points = data.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    return distance_matrix


# Function to calculate local density (rho)
def calculate_rho(distance_matrix, dc):
    num_points = distance_matrix.shape[0]
    rho = np.zeros(num_points)
    for i in range(num_points):
        rho[i] = np.sum(np.exp(-(distance_matrix[i] / dc) ** 2))
    return rho


# Function to calculate delta and nearest higher density point
def calculate_delta(distance_matrix, rho):
    num_points = distance_matrix.shape[0]
    delta = np.zeros(num_points)
    nearest_higher_density = np.zeros(num_points, dtype=int)
    for i in range(num_points):
        higher_density_points = np.where(rho > rho[i])[0]
        if len(higher_density_points) > 0:
            delta[i] = np.min(distance_matrix[i, higher_density_points])
            nearest_higher_density[i] = higher_density_points[np.argmin(distance_matrix[i, higher_density_points])]
        else:
            delta[i] = np.max(distance_matrix[i])
            nearest_higher_density[i] = i
    return delta, nearest_higher_density


# Function to find cluster centers
def find_cluster_centers(rho, delta, rho_threshold, delta_threshold):
    return np.where((rho > rho_threshold) & (delta > delta_threshold))[0]


# Function to assign clusters
def assign_clusters(distance_matrix, nearest_higher_density, cluster_centers, rho):
    num_points = distance_matrix.shape[0]
    labels = -np.ones(num_points, dtype=int)
    for i, center in enumerate(cluster_centers):
        labels[center] = i
    sorted_rho_indices = np.argsort(-rho)
    for i in sorted_rho_indices:
        if labels[i] == -1:
            labels[i] = labels[nearest_higher_density[i]]
    return labels


# Neural Network for optimizing rho and delta thresholds
class CFSDPNet(nn.Module):
    def __init__(self):
        super(CFSDPNet, self).__init__()
        self.rho_threshold = nn.Parameter(torch.tensor(0.5))
        self.delta_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, data):
        distance_matrix = calculate_distance_matrix(data)
        dc = np.percentile(distance_matrix, 2)
        rho = calculate_rho(distance_matrix, dc)
        delta, nearest_higher_density = calculate_delta(distance_matrix, rho)

        cluster_centers = find_cluster_centers(rho, delta, self.rho_threshold.item(), self.delta_threshold.item())
        labels = assign_clusters(distance_matrix, nearest_higher_density, cluster_centers, rho)

        return labels


# Function to train the model
def train_model(data):
    model = CFSDPNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Fake target for demonstration (randomly assigning clusters)
    fake_target = torch.tensor(np.random.randint(0, 3, data.shape[0]), dtype=torch.float)

    for epoch in range(100):
        logging.info("epoch: {}".format(epoch))
        optimizer.zero_grad()
        output_labels = model(data)
        output_labels_tensor = torch.tensor(output_labels, dtype=torch.float)

        loss = criterion(output_labels_tensor, fake_target)
        logging.info("loss: {}".format(loss))
        loss.backward()
        optimizer.step()

    return model


# Function to extract keywords
def extract_keywords(text, model, num_keywords=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text]).toarray()
    words = np.array(vectorizer.get_feature_names_out())

    data = torch.tensor(tfidf_matrix, dtype=torch.float32)
    final_labels = model(data)

    keyword_indices = np.argsort(-final_labels)[:num_keywords]
    keywords = words[keyword_indices]

    return keywords


# Example usage
if __name__ == "__main__":
    # Load dataset for training
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(newsgroups.data).toarray()

    # Convert to torch tensor
    data = torch.tensor(tfidf_matrix, dtype=torch.float32)

    # Train model
    model = train_model(data)

    # Extract keywords from a new text
    text = "Machine learning is a method of data analysis that automates analytical model building. It is a branch of " \
           "artificial intelligence based on the idea that systems can learn from data, identify patterns and make " \
           "decisions with minimal human intervention. "
    keywords = extract_keywords(text, model)
    print("Keywords:", keywords)
