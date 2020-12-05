import numpy as np
import random
import operator 
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Centroid():
    
    def __init__(self, cl, acc):
        self.cl = cl
        self.acc = acc
        self.count = 1

    def append(self, data):
        for i, val in enumerate(self.acc):
            self.acc[i] += data[i]
            self.count += 1

    def getCentroid(self):
        return self.acc / self.count


# Reads and normalize the database, returns the data and classes apart
def readDatabase(filename, has_id, class_position):
    
    filepath = os.path.join("databases",filename)
    
    with open(filepath) as f:
        # Getting only the lines without missing attribute
        lines = (line for line in f if '?' not in line)
        dataset = np.loadtxt(lines, delimiter = ',')

    # Shuffing the dataset, once sometimes the data are grouped by class
    np.random.shuffle(dataset)

    # Considering the last column being the class column
    if class_position == 'first':
        classes = dataset[:, 0]
        dataset = np.delete(dataset, 0, axis = 1)
    else:   
        classes = dataset[:, -1]
        dataset = np.delete(dataset, -1, axis = 1)

    if has_id:
        # Remove the first column (ID)
        dataset = np.delete(dataset, 0, axis = 1)

    # Normalizing the data in the [0 1] interval
    arr_max = np.max(dataset, axis = 0) # gets the max of each column
    arr_min = np.min(dataset, axis = 0) # gets the min of each column

    rows, cols = np.shape(dataset)
    for i in range(rows):
        for j in range(cols):
            dataset[i][j] = (dataset[i][j] - arr_min[j]) / (arr_max[j] - arr_min[j])

    return dataset, classes

# Determine the classes centroids as the mean values of the data
# in each class
def determineCentroids(dataset, classes):
    
    rows, cols = np.shape(dataset)

    stats = {}

    for i, row in enumerate(dataset):
        class_id = str(classes[i])
        if class_id in stats:
            stats[class_id].append(row)
        else:
            stats[class_id] = Centroid(classes[i], row)

    centroids = {}
    for key in stats:
        centroids[key] = stats[key].getCentroid()

    return stats, centroids

# Simple Euclidian distance between two arrays
def euclidianDistance(a, b):    
    
    diff_sqrt = [(x - y)**2 for x, y in zip(a, b)]

    return np.sqrt(np.sum(diff_sqrt))

# The sum of the distances between a data point and its class centroid
# in the trainning set
def costFunction(dataset, classes, cl, centroid):
    
    # 'cl' will be the string representation of the class already
    distances_sum = 0
    count = 0
    for i, d in enumerate(dataset):
        if str(classes[i]) == cl: # limiting the search only in the specific class
            distances_sum += euclidianDistance(d, centroids[cl])
            count += 1

    return distances_sum / count

def fitnessFunction(costs):
    
    fitness = costs.copy()
    for key in fitness:
        fitness[key] = 1/(1 + costs[key])

    return fitness

def fitnessVal(cost):
    return 1/(1+cost)

def rouletteWheelFunction(P):
    p_sorted_asc = sorted(P.items(), key = operator.itemgetter(1))
    p_sorted_desc = dict(reversed(p_sorted_asc))

    pick = np.random.uniform(0, 1)
    current = 0
    for key in p_sorted_desc:
        current += p_sorted_desc[key]
        if current > pick:
            return key



# Artificial Bee Colony algorithm implementation
def ABC(dataset, classes, centroids, a_limit, max_iter):
    
    n_data, n_attr = np.shape(dataset) # Number of cases and number of attributes in each case
    n_bees = len(centroids) # Number of bees in the problem
    var_min = 0 # Minimum possible for each variable
    var_max = 1 # Maximum possible for each variable

    keys = [key for key in centroids] # centroid keys

    # Initialize the counter of rejections array
    C = centroids.copy()
    for key in C:
        C[key] = 0

    # Initilize the cost array
    costs = centroids.copy()
    for cl in costs:
        costs[cl] = costFunction(dataset, classes, cl, centroids[cl])

    best_solution = 99999999
    best_solutions = np.zeros(max_iter)

    for it in range(max_iter):
        # Employed bees phase

        F = fitnessFunction(costs) # calculate fitness of each class

        for cl in centroids:
            _keys = keys.copy() # copying to maintain the original dict
            index = _keys.index(cl)
            del _keys[index]
            k = random.choice(_keys) # getting a index k different from i

            # Define phi coefficient to generate a new solution
            phi = np.random.uniform(-1, 1, n_attr)

            # Generating new solution
            # centroids: numpy array
            # phi: numpy array
            # (centroids[cl] - centroids[k]): numpy array
            # The operation will be element by element given that all the operands
            # are numpy arrays
            new_solution = centroids[cl] + phi * (centroids[cl] - centroids[k])

            # Calculate the cost of the dataset with the new centroid
            new_solution_cost = costFunction(dataset, classes, cl, new_solution)

            new_fitness = fitnessVal(new_solution_cost)

            # Greedy selection: comparing the new solution to the old one
            if new_fitness > F[cl]:
                centroids[cl] = new_solution
                F[cl] = new_solution_cost
                # F[cl] = new_fitness
                # costs[cl] = new_solution_cost
                C[cl] = 0
            else: 
                # Increment the counter for discarted new solutions
                C[cl] += 1

        f_sum_arr = [F[key] for key in F]
        f_sum = np.sum(f_sum_arr)
        
        P = {} # probabilities of each class
        for key in F:
            P[key] = F[key]/f_sum

        # Onlooker bees phase
        for cl_o in centroids:
            selected_key = rouletteWheelFunction(P)

            _keys = keys.copy() # copying to maintain the original dict
            index = _keys.index(selected_key)
            del _keys[index]
            k = random.choice(_keys) # getting a index k different from i

            # Define phi coefficient to generate a new solution
            phi = np.random.uniform(-1, 1, n_attr)

            # Generating new solution
            # centroids: numpy array
            # phi: numpy array
            # (centroids[selected_key] - centroids[k]): numpy array
            # The operation will be element by element given that all the operands
            # are numpy arrays
            new_solution = centroids[selected_key] + phi * (centroids[selected_key] - centroids[k])

            # Calculate the cost of the dataset with the new centroid
            new_solution_cost = costFunction(dataset, classes, selected_key, new_solution)

            new_fitness = fitnessVal(new_solution_cost)

            # Greedy selection: comparing the new solution to the old one
            if new_fitness > F[selected_key]:
                centroids[selected_key] = new_solution
                F[selected_key] = new_solution_cost
                # F[selected_key] = new_fitness
                # costs[selected_key] = new_solution_cost
                C[selected_key] = 0
            else: 
                # Increment the counter for discarted new solutions
                C[selected_key] += 1

        max_fails = a_limit
        c_cl = None

        # Scout bees phase
        for cl_s in centroids:
            if C[cl_s] > a_limit:
                if C[cl_s] > max_fails:
                    max_fails = C[cl_s]
                    c_cl = cl_s

        if c_cl:
            cl_s = c_cl
            random_solution = np.random.uniform(0, 1, n_attr)
            random_solution_cost = costFunction(dataset, classes, cl_s, random_solution)

            new_fitness = fitnessVal(random_solution_cost)

            centroids[cl_s] = new_solution
            F[cl_s] = new_fitness
            C[cl_s] = 0

        # Update best solution for this iteration
        best_solution = 9999999999
        for cl in centroids:
            if costs[cl] < best_solution:
                best_solution = costs[cl]

        best_solutions[it] = best_solution

    return best_solutions, centroids

def nearestCentroidClassifier(data, centroids):
    
    distances = centroids.copy()
    for key in centroids:
        distances[key] = euclidianDistance(data, centroids[key])

    distances_sorted = sorted(distances.items(), key = operator.itemgetter(1))
    nearest_class, nearest_centroid = distances_sorted[0]

    return nearest_class

def getSets(dataset, classes):
    
    size = len(dataset)

    trainning_set = dataset[:round(size * 0.75), :]
    trainning_set_classes = classes[:round(size * 0.75)]

    test_set = dataset[round(size * 0.75):, :]
    test_set_classes = classes[round(size * 0.75):]

    return trainning_set, test_set, trainning_set_classes, test_set_classes



databases = [
            { 'filename': 'cancer_int.data', 'has_id': True, 'class_position': 'last' }, 
            { 'filename': 'cancer.data', 'has_id': True, 'class_position': 'last' },
            { 'filename': 'new-thyroid.data', 'has_id': False, 'class_position': 'first' },
            { 'filename': 'heart_processed.data', 'has_id': False, 'class_position': 'last' },
            { 'filename': 'haberman.data', 'has_id': False, 'class_position': 'last' },
            { 'filename': 'lymphography.data', 'has_id': False, 'class_position': 'first' }
            ]

for database in databases:
    
    d, c = readDatabase(database['filename'], database['has_id'], database['class_position'])
    trainning_set, test_set, trainning_set_classes, test_set_classes = getSets(d.copy(), c.copy())

    print('\n\n## DATABASE: {filename}'.format(filename = database['filename']))

    stats, centroids = determineCentroids(trainning_set, trainning_set_classes)

    clusters = len(centroids)

    # KMeans CLustering
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(trainning_set)

    classes_labels = {}
    pred_test = kmeans.predict(test_set)

    for y_label in pred_test:
        classes_labels[y_label] = {}

    for pred_label, test_label in zip(pred_test, test_set_classes):
        if classes_labels[pred_label].get(test_label) is None:
            classes_labels[pred_label][test_label] = 1
        else:
            classes_labels[pred_label][test_label] += 1

    correct_count = 0
    for key, val in classes_labels.items():
        max_inst = 0
        for k,v in classes_labels[key].items():
            if max_inst < v:
                max_inst = v
        correct_count += max_inst
    acc = correct_count / len(pred_test)

    print("KMEANS ACCURACY: ", acc)

    max_acc = 0
    best_limit = -1

    ######## ABC OUTPUT ########
    limits = [1000, 500, 50]
    for limit in limits:
        best_soltions, new_centroids = ABC(trainning_set, trainning_set_classes, centroids.copy(), a_limit = limit, max_iter = 5000)

        # Calculate Test Accuracy with the ABC result
        count = 0

        for i, val in enumerate(test_set):
            cl = nearestCentroidClassifier(test_set[i], new_centroids)
            if cl == str(test_set_classes[i]):
                count += 1
        
        abc_acc = count/len(test_set)

        if max_acc < abc_acc:
            max_acc = abc_acc
            best_limit = limit
    
    print("\n\nTest with the ABC result centroids")
    print(f"# BEST ABC RESULT at limit {best_limit} -> ACCURACY: {max_acc}\n")