import math

import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings = model.encode(documents)

print(f"Let's explore the shape of our embeddings: {embeddings.shape}")
print(f"let's have a quick look at them: {len(embeddings)}")

def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x-y) ** 2 for x,y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

print(f"let's calculate the distance between the first vector (index 0) and the second vector (index 1): {euclidean_distance_fn(embeddings[0], embeddings[1])}")

l2_dist_manual = np.zeros([4,4])

for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        if i == j:
            continue
        else:
            if l2_dist_manual[j,i] != 0:
                l2_dist_manual[i,j] = l2_dist_manual[j,i]
            else:
                l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i],embeddings[j])
# more efficient one
# for i in range(embeddings.shape[0]):
#     for j in range(embeddings.shape[0]):
#         if j > i: # Calculate the upper triangle only
#             l2_dist_manual_improved[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
#         elif i > j: # Copy the uper triangle to the lower triangle
#             l2_dist_manual_improved[i,j] = l2_dist_manual[j,i]
print(f"Distances:\n{l2_dist_manual}")


l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print(f"let's have a quick look at l2_dist_scipy: \n{l2_dist_scipy}")

print(f"The following verifies that l2_dist_manual and l2_dist_scipy are identical: {np.allclose(l2_dist_manual, l2_dist_scipy)}")

def dot_product_fn(vector1, vector2):
    return sum(x * y for x,y in zip(vector1, vector2))

print(f"let's calculate the dot product between the first vector (index 0) and the second vector (index 1): {dot_product_fn(embeddings[0], embeddings[1])}")

dotProduct_dist_manual = np.zeros([4,4])

# for i in range(len(embeddings)):
#     for j in range(len(embeddings)):
#         dotProduct_dist_manual[i,j] = dot_product_fn(embeddings[i],embeddings[j])
# more efficient one
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j >= i: # Calculate the upper triangle only
            dotProduct_dist_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])
        elif i > j: # Copy the uper triangle to the lower triangle
            dotProduct_dist_manual[i,j] = dotProduct_dist_manual[j,i]
print(f"dot product Distances:\n{dotProduct_dist_manual}")

dot_product_operator = embeddings @ embeddings.T

print(f"dot product with @ operator:\n{dot_product_operator}")

print(f"The following verifies that dotProduct_dist_manual and dot_product_operator are identical: {np.allclose(dotProduct_dist_manual, dot_product_operator, atol=1e-05)}")

print(f"The following provides dot product using matmul: \n{np.matmul(embeddings,embeddings.T)}")
print(f"The following provides dot product using np.dot: \n{np.dot(embeddings,embeddings.T)}")

dot_product_distance = -dotProduct_dist_manual
print(f"The following provides distances using dot product: \n{dot_product_distance}")