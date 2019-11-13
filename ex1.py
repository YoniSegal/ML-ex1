# Yonatan Segal 342476611
import numpy as np
from scipy.misc import imread
from init_centroids import init_centroids


# Method returns the closest centroid to a given pixel.
def closest_centroid(p, centroids):
    distances = []
    for c in centroids:
        distances.append(np.linalg.norm(p - c) ** 2)
    min_index = np.argmin(distances)
    for i in range(len(centroids)):
        if np.linalg.norm(p - centroids[i]) ** 2 == distances[min_index]:
            return centroids[i]


# Method assigns each pixel to a given centroid.
def assign_groups(X, centroids, centroid_sets):
    # centroid_sets 1..n -> centroids 1..n
    # for each point x, if centroids[j] is closest to x, put x into centroid_sets[j]
    for p in X:
        closest = closest_centroid(p, centroids)
        i = 0
        for j in range(len(centroids)):
            if (centroids[j] == closest).all():
                centroid_sets[i].append(p)
                break
            i += 1


# Method divides 2 3D vectors.
def div_vector(total_centroid, size):
    return [total_centroid[0] / size, total_centroid[1] / size, total_centroid[2] / size]


# Method returns a new array of centroids based on the average pixel values.
def update_centroids(centroid_sets):
    new_centroids = []
    total_centroid = [0., 0., 0.]
    for cluster in centroid_sets:
        for colour in cluster:
            total_centroid += colour
        average_centroid = div_vector(total_centroid, len(cluster))
        new_centroids.append(average_centroid)
        total_centroid = [0., 0., 0.]
    return np.asarray(new_centroids)


# Method prints all centroids in a given iteration.
def print_centroids(centroids, i, k):
    final_string = "iter " + str(i) + ": "
    string = "["
    counter = 0
    for c in centroids:
        first = str(np.floor((c[0] * 100)) / 100)
        second = str(np.floor((c[1] * 100)) / 100)
        third = str(np.floor((c[2] * 100)) / 100)
        if first == "0.0":
            first = "0."
        if second == "0.0":
            second = "0."
        if third == "0.0":
            third = "0."
        counter += 1
        string += first + ", " + second + ", " + third + "]"
        if counter != k:
            string += ", ["
    final_string += string
    print(final_string)


# Main method takes an image and uses k-means to alter pixels around given centroids.
def main():
    path = 'dog_2.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    k = [2, 4, 8, 16]
    for k in k:
        centroids = init_centroids(X, k)
        print("k=" + str(k) + ":")
        print_centroids(centroids, 0, k)
        for i in range(10):
            # centroid_sets 1..n -> centroids 1..n
            # for each point x, if centroids[i] is closest to x, put x into centroid_sets[i]
            centroid_sets = [[] for j in range(k)]
            assign_groups(X, centroids, centroid_sets)
            centroids = update_centroids(centroid_sets)
            # print
            print_centroids(centroids, i + 1, k)


main()
