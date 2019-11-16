# Yonatan Segal 342476611
import sys

import numpy as np
from scipy.io import wavfile


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


# Method divides 2 2D vectors.
def div_vector(total_centroid, size):
    # return [total_centroid[0] / size, total_centroid[1] / size]
    return [round(total_centroid[0] / size), round(total_centroid[1] / size)]


# Method returns a new array of centroids based on the average pixel values.
def update_centroids(centroid_sets):
    new_centroids = []
    total_centroid = [0., 0.]
    for cluster in centroid_sets:
        for sound in cluster:
            total_centroid += sound
        average_centroid = div_vector(total_centroid, len(cluster))
        new_centroids.append(average_centroid)
        total_centroid = [0., 0.]
    return np.asarray(new_centroids)


# Main method takes an image and uses k-means to alter pixels around given centroids.
if __name__ == '__main__':
    wav = sys.argv[1]
    cents = sys.argv[2]
    text_file = open("output2.txt", "w")
    fs, y = wavfile.read(wav)
    centroids = np.loadtxt(cents)
    k = [int(centroids.size / 2)]
    previous = ""
    for k in k:
        for iter in range(30):
            # centroid_sets 1..n -> centroids 1..n
            # for each point x, if centroids[i] is closest to x, put x into centroid_sets[i]
            centroid_sets = [[] for j in range(k)]
            assign_groups(y, centroids, centroid_sets)
            centroids = update_centroids(centroid_sets)
            temp1 = "{}".format(','.join(str(i) for i in centroids))
            print("[iter {}]:".format(iter) + "{}".format(','.join(str(i) for i in centroids)))
            text_file.write("[iter {}]:".format(iter) + "{}".format(','.join(str(i) for i in centroids)) + "\n")
            if iter > 0 and temp1 == previous:
                break
            previous = "{}".format(','.join(str(i) for i in centroids))
    text_file.close()
