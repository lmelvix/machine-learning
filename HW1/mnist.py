import os
import struct
from array import array as pyarray
from pylab import *
from numpy import *
from numpy import array, int8, uint8, zeros
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import warnings


def load_mnist(dataset="training", digits=np.arange(10), path=""):
    '''
    Load Binary MNIST dataset images and corresponding labels
    :param dataset: Type of dataset to be loaded
    :param digits:  Range of digits to be stored
    :param path:  Path to MNIST dataset
    :return: Images and corresponding labels
    '''
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')

    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
            images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
            labels[i] = lbl[ind[i]]

    return images, labels


def calculate_distance(image1,  image2):
    return distance.euclidean(image1, image2)


def prototype_select(train_images, train_labels, prototype_size, technique="random"):
    prototype_image = np.empty((1, train_images.shape[1]), int)
    prototype_label = np.empty((1, train_labels.shape[1]), int)

    if technique == "random":
        train_images, train_labels = load_mnist('training', digits=[1,2,3,4,5,6,7,8,9,0])
        for loop in range(prototype_size-1):
            rand_index = randint(0,train_labels.shape[0]-1)
            prototype_image = np.vstack((prototype_image, train_images[rand_index]))
            prototype_label = np.vstack((prototype_label, train_labels[rand_index]))

    elif technique == "mean_close":
        for digit in range(0,9):
            prototype_dict = {}
            train_images,train_labels = load_mnist('training', digits = np.array([digit]))
            mean_image = train_images.mean(axis=0)
            prototype_image = np.vstack((prototype_image, mean_image))
            prototype_label = np.vstack((prototype_label, train_labels[0]))

            for image in train_images:
                prototype_dict[calculate_distance(mean_image, image)] = image
            dict_distance =  sorted(prototype_dict)[:((int)(prototype_size/10)-1)]
            for dic_dist in dict_distance:
                prototype_image = np.vstack((prototype_image, prototype_dict[dic_dist]))
                prototype_label = np.vstack((prototype_label, train_labels[0]))

    elif technique == "mean_outlier":
        for digit in range(0,9):
            prototype_dict = {}
            train_images,train_labels = load_mnist('training', digits = np.array([digit]))
            mean_image = train_images.mean(axis=0)
            prototype_image = np.vstack((prototype_image, mean_image))
            prototype_label = np.vstack((prototype_label, train_labels[0]))

            for image in train_images:
                prototype_dict[calculate_distance(mean_image, image)] = image
            dict_distance =  sorted(prototype_dict, reverse = True)[:((int)(prototype_size/10)-1)]
            for dic_dist in dict_distance:
                prototype_image = np.vstack((prototype_image, prototype_dict[dic_dist]))
                prototype_label = np.vstack((prototype_label, train_labels[0]))

    elif technique == "mean_hop":
        for digit in range(0,9):
            prototype_dict = {}
            train_images,train_labels = load_mnist('training', digits = np.array([digit]))
            hop_count = int((train_images.shape[0]*10)/prototype_size)
            mean_image = train_images.mean(axis=0)
            prototype_image = np.vstack((prototype_image, mean_image))
            prototype_label = np.vstack((prototype_label, train_labels[0]))

            for image in train_images:
                prototype_dict[calculate_distance(mean_image, image)] = image
            dict_distance =  sorted(prototype_dict, reverse = True)[:((int)(prototype_size/10)-1)]
            for index in range(0,len(dict_distance)-1,hop_count):
                prototype_image = np.vstack((prototype_image, prototype_dict[dict_distance[index]]))
                prototype_label = np.vstack((prototype_label, train_labels[0]))

    elif technique == "hop_cluster":
        for digit in range(0,9):
            prototype_dict = {}
            train_images,train_labels = load_mnist('training', digits = np.array([digit]))
            hop_count = int((train_images.shape[0]*10)/prototype_size)
            mean_image = train_images.mean(axis=0)
            prototype_image = np.vstack((prototype_image, mean_image))
            prototype_label = np.vstack((prototype_label, train_labels[0]))

            for image in train_images:
                prototype_dict[calculate_distance(mean_image, image)] = image
            dict_distance =  sorted(prototype_dict, reverse = True)[:((int)(prototype_size/10)-1)]
            for index in range(0,len(dict_distance)-1,hop_count):
                cluster_image = np.empty((1, train_images.shape[1]), int)
                for range_index in range(index-hop_count,index+hop_count):
                    if range_index > 0 and range_index < len(dict_distance):
                        cluster_image = np.vstack(prototype_dict[dict_distance[range_index]])
                cluster_image = cluster_image.mean(axis=1)
                prototype_image = np.vstack((prototype_image, cluster_image))
                prototype_label = np.vstack((prototype_label, train_labels[0]))

    return prototype_image,prototype_label


def evaluate_classifier(train_images, train_labels):
    test_images, test_labels = load_mnist('testing', digits = [1,2,3,4,5,6,7,8,9,0])
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_images, train_labels.ravel())
    count = 0.0
    error = 0.0
    for index in range(test_labels.size):
        predict = neigh.predict(test_images[index])
        answer = test_labels[index]
        if predict != answer:
            error += 1.0
        count += 1.0
    return str((error/count)*100) + "%"


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    train_images, train_labels = load_mnist('training', digits=[1,2,3,4,5,6,7,8,9,0])
    #  Baseline Classifier with complete Training set
    # error = evaluate_classifier(train_images, train_labels)
    prototype_size = [1000,3000,5000,7000,10000,15000,20000]
    select_algorithm = ["random", "mean_close", "mean_outlier",
                        "mean_hop", "hop_cluster"]
    for M in prototype_size:
        for algo in select_algorithm:
            train_image_prototype, train_label_prototype = \
                prototype_select(train_images, train_labels, M, algo)
            error = evaluate_classifier(train_image_prototype,
                                        train_label_prototype)
            with open("1NN_Result.txt", "a") as myfile:
                result = str(M) + "\t\t" + \
                         str(algo) + "\t\t" + str(error) +"\n"
                print result
                myfile.write(result)

if __name__ == "__main__" :
    main()
