import pandas as pd # reading all required header files
import numpy as np
import random
import operator
import math
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min

##############################################################################
# Load Dataset
##############################################################################

df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\FINAL YEAR PROJECT\\MY_PROJECT\\FOR IRIS DATASET\\IRIS.csv")

#print(df)
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'species' column
df['species'] = label_encoder.fit_transform(df['species'])
#print(df)
target=df.iloc[:,-1]
#print(target)
df= df.drop(columns=['species']) # Class labels
#print(df)
target = pd.DataFrame(target)
_,col_num = df.shape
#print(df)
#print(df.columns)

#print(df.head())
##############################################################################
#number of data
n = len(df)
#number of clusters
k = 3
#dimension of cluster
d = col_num
# m parameter
m = 2
#number of iterations
MAX_ITERS = 12
##############################################################################

##############################################################################
# Initializing Membership Matrix
# wij values are assigned randomly.
##############################################################################
def initializeMembershipWeights():
  weight = np.random.dirichlet(np.ones(k),n)
  weight_arr = np.array(weight)
  return weight_arr

##############################################################################
# Calculating Cluster Center
# To calculate centroids for each cluster we apply the following formula:
# Cj and m(fuzzy-ness) ranges from 1 to infinity
##############################################################################

def computeCentroids(weight_arr,df):
  C = []
  for i in range(k):
    weight_sum = np.power(weight_arr[:,i],m).sum()
    Cj = []
    for x in range(d):
      numerator = ( df.iloc[:,x].values * np.power(weight_arr[:,i],m)).sum()
      c_val = numerator/weight_sum;
      Cj.append(c_val)
    C.append(Cj)
  return C 


##############################################################################
# Updating Membership Value
# Calculate the fuzzy-pseudo partition with the above formula
# $$w_{ij} = \frac{(\frac{1}{dist(x_i, c_j)})^{\frac{1}{m-1}}}{\sum_{s=1}^{k}(\frac{1}{dist(x_i,c_s)})^{\frac{1}{m-1}}}w
# ij$$
# 
# where 
#  dist(x_i, c_j) is the Euclidean distance between x_i and c_j cluster center.
##############################################################################


def updateWeights(weight_arr,C,df):
  denom = np.zeros(n)
  for i in range(k):
    dist = (df.iloc[:,:].values - C[i])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    denom  = denom + np.power(1/dist,1/(m-1))

  for i in range(k):
    dist = (df.iloc[:,:].values - C[i])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    weight_arr[:,i] = np.divide(np.power(1/dist,1/(m-1)),denom)
  return weight_arr
##############################################################################
# finding the highest membership and next highest (2nd highest) membership value
##############################################################################
def findHighestAndNextHighestElement(m):
    #Declaring arrays like highest, nextHighest, diff
    highest = np.zeros(len(m))
    nextHighest = np.zeros(len(m))
    diff = np.zeros(len(m))
    m = np.sort(m)
    for i in range(len(m)):
        highest[i] = m[i][len(m[i])-1]
        nextHighest[i] = m[i][len(m[i])-2]
        diff[i] = highest[i] - nextHighest[i]
    return (highest, nextHighest, diff)

##############################################################################
# Fuzzy algorithm
#############################################################################
def FuzzyMeansAlgorithm(df):
    weight_arr = initializeMembershipWeights()
    for z in range(MAX_ITERS):
        C = computeCentroids(weight_arr,df)
        updateWeights(weight_arr,C,df)
 
    return (weight_arr,C)

#calling the FuzzyMeansAlgorithm.
final_weights,Centers = FuzzyMeansAlgorithm(df)
#Finding highest and next highest membership value
highest, nextHighest, diff = findHighestAndNextHighestElement(final_weights)
min_diff = min(diff)  # rename min variable
max_diff = max(diff)

############################################################################
#Call from optimize_threshold value, where we find the core boundary value
############################################################################
def find_core_boundary(diff,final_weights,df,target,threshold):
    
    if(threshold==max_diff):
   
        indices_core_value = [i for i in range(n) if diff[i] >= threshold]
        indices_boundary_value = [i for i in range(n) if diff[i] < threshold]
    else:
        indices_core_value = [i for i in range(n) if diff[i] > threshold]
        indices_boundary_value = [i for i in range(n) if diff[i] <= threshold]
        
    df_core_value = df.iloc[indices_core_value]
    df_core_cluster=target.iloc[indices_core_value]
    df_boundary_value = df.iloc[indices_boundary_value]
  
    dataframe_core_value = pd.DataFrame(df_core_value, columns=df.columns)
    dataframe_core_cluster = pd.DataFrame(df_core_cluster)
    dataframe_boundary_value = pd.DataFrame(df_boundary_value, columns=df.columns)
       
    core_data = dataframe_core_value.iloc[:, :col_num] # it store the first fourth column of core values.
       
    core_cluster = dataframe_core_cluster.iloc[:, 0] # it store the only the cluster column.
       
    boundary_data = dataframe_boundary_value.iloc[:, :col_num] ## it store the first fourth column pf boundary values.
       
    return (core_data,core_cluster,boundary_data)
    

############################################################################
#we apply the supervised lerning (KNN Classification ) to classifi the boundary point
#############################################################################
def KNN(core_data, core_cluster, boundary_data):
    model = KNeighborsClassifier(n_neighbors=min(len(core_data), k))  # Use the minimum of k and the number of core_data samples
    model.fit(core_data.values, core_cluster)

    boundary_cluster = []
    if len(boundary_data) > 0:
        for index, row in boundary_data.iterrows():
            new_data = [row.tolist()]  # Convert row to list of values
            prediction = model.predict(new_data)[0]  # Get the prediction
            boundary_cluster.append(prediction)  # Append prediction to the list
    else:
        # Handle the case where boundary_data is empty
        boundary_cluster = []

    # Concatenate the dataframes along columns
    concat_df = pd.concat([core_data, boundary_data], ignore_index=True)

    # Convert to 2D NumPy array with data type preservation
    new_df = concat_df.to_numpy()
    #new_dataframe = pd.DataFrame(new_df)
    #final_weights,Centroid = FuzzyMeansAlgorithm(new_dataframe)
    #print(Centers)
    # Convert the boundary_cluster list into a DataFrame
    boundary_cluster_df = pd.DataFrame(boundary_cluster)

    # Concatenate the dataframes along columns
    cluster = pd.concat([core_cluster, boundary_cluster_df], ignore_index=True)

    # Convert the combined_cluster into an array
    clusters = cluster.to_numpy()

    return clusters, new_df

##############################################################################
#  compute the xb index value
##############################################################################

def compute_xb_index(data, labels, centroids):
    k = len(centroids)
    n = len(data)
    distances = pairwise_distances_argmin_min(data, centroids)[1]	# Calculate the distance matrix
    within_cluster_distances = np.sum(distances) # Calculate the sum of the within-cluster distances
    cluster_dispersion = 0
    for i in range(k):
        cluster_points = data[np.where(labels == i)[0]]
        if cluster_points.shape[0] == 0:                   # Check for empty cluster and skip if necessary
            continue
        cluster_dispersion += np.sum(pairwise_distances_argmin_min(cluster_points, [centroids[i]])[1])
    xie_beni = within_cluster_distances / (k * cluster_dispersion / n)     # Calculate the Xie-Beni index
    return xie_beni
##############################################################################
#call from fitness to find the optimize value
##############################################################################
def optimize_threshold(threshold):
    # Call find_core_boundary function with the provided threshold
    core_data, core_cluster, boundary_data = find_core_boundary(diff, final_weights, df, target,threshold)
    
    clustering, new_df = KNN(core_data, core_cluster, boundary_data)  # Assuming KNN returns clustering and new_df
    xb_index = compute_xb_index( new_df,clustering,Centers)

    
    return xb_index
##############################################################################
#Start of Genetic Algorithm
##############################################################################
##############################################################################
#initialize the population 
##############################################################################
def initialize_population(population_size_param, chromosome_length_param):
    population = []
    for i in range(population_size_param):
        individual = [random.choice([0, 1]) for i in range(chromosome_length_param)]
        population.append(individual)
    
    return population
#############################################################################
# decode the the chromosome
#############################################################################
def decode_chromosome(chromosome):
    binary_string = ''.join(map(str, chromosome))
    decimal_value = int(binary_string, 2)
    return decimal_value
############################################################################
#Calculate the thresold value and with in define range
############################################################################
def scale_numbers(number, target_min, target_max):
    # Define the original range
    original_min = 0
    original_max = 255
    # Scale the number
    scaled_number = ((number - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min
    return scaled_number
#############################################################################
#Calculate the fitness of each individuals
#############################################################################
def fitness_ev(individual):
    dec = decode_chromosome(individual)
    threshold = scale_numbers(dec, target_min=min_diff, target_max=max_diff)
    xb_index=(optimize_threshold(threshold))
    return xb_index 
############################################################################
#elect the fitted parents using Binary Tournament selection process
############################################################################
def select_parents(population,fitness):        # Binary Tournament Selection
    matpool = np.zeros_like(population, dtype=int)
    for i in range(len(population)):
        # Pick two random candidates between 0 and population_size - 1
        cand1 = np.random.randint(0, len(population))
        cand2 = np.random.randint(0, len(population))
        if fitness[cand1] <= fitness[cand2]:
            selected = cand1
        else:
            selected = cand2
        matpool[i] = np.copy(population[selected])
    return matpool
#############################################################################
# Uniform Crossover
#############################################################################
def crossover(parent1, parent2, crossover_rate):    
    if np.random.rand() < crossover_rate:
        # If crossover occurs, create a random mask
        mask = np.random.randint(0, 2, size=len(parent1))
        # Initialize child chromosomes
        child1 = np.zeros(len(parent1), dtype=int)
        child2 = np.zeros(len(parent2), dtype=int)
        for i in range(len(parent1)):
            if mask[i] == 0:  # No change if mask bit is 0
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:              # Exchange if mask bit is 1
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        return child1, child2
    else:
        # If no crossover, return the parents unchanged
        return parent1, parent2
################################################################################
#Mutation to mutate the chromosome
################################################################################
def mutate(individual, mutation_rate):
    mutated_individual = [gene if random.random() > mutation_rate else 1 - gene for gene in individual]
    return mutated_individual
################################################################################
#Genetic algorithm
################################################################################
def genetic_algorithm(population_size_param, chromosome_length_param, mutation_rate_param, crossover_rate_param, generations_param, consecutive_generations_threshold):
    population = initialize_population(population_size_param, chromosome_length_param)
    best_fitness_values = []
    fitness=[]
    best_individual = None  # Initialize best individual as None
    best_fitness = float('inf')
    consecutive_generations = 0
    for generation in range(generations_param):
        fitness = [fitness_ev(chromosome) for chromosome in population]
         # Select parents from the remaining population
        parents = select_parents(population,fitness)
        # Perform crossover and mutation
        offspring_population = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[i]
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate_param)
            offspring_population.append(mutate(offspring1, mutation_rate_param))
            offspring_population.append(mutate(offspring2, mutation_rate_param))
        # Combine elite and offspring populations
        population = offspring_population
        # Update best fitness and best individual
        min_fitness = min(fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[fitness.index(min_fitness)]
            consecutive_generations = 0  # Reset consecutive generations counter
        else:
            consecutive_generations += 1
        # Append best fitness of this generation
        best_fitness_values.append(best_fitness)
        # Find the index of the current best chromosome
        best_index = fitness.index(min(fitness))
        best_chromosome = population[best_index]
       # Find the elite chromosome and its index (initially the previous best)
        elite_index = fitness.index(min(fitness[:best_index] + fitness[best_index + 1:]))
        elite_chromosome = population[elite_index]
         # Check if the current best is better than the previous elite
        if fitness[best_index] > fitness[elite_index]:
            elite_chromosome = best_chromosome
            elite_index = best_index
        new_population = population.copy()
        # Check if the (potentially updated) elite chromosome is already present
        if elite_chromosome not in new_population:
        # Find the index of the worst chromosome (lowest fitness) if needed
            worst_index = fitness.index(max(fitness))
            new_population[worst_index] = elite_chromosome
        population=new_population
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        # Check if consecutive_generations_threshold has been reached
        if consecutive_generations >= consecutive_generations_threshold:
            print(f"Stopping criteria met: {consecutive_generations_threshold} consecutive generations with the same best fitness value.")
            break;
    best_fitness = min(best_fitness_values)
    ind=best_fitness_values.index(best_fitness)
    best_individual=vidual=population[ind]
    print("index",ind)
    print("Final Population:")
    for individual in population:
        print(individual)     

    # Return the best fitness values, best fitness, and best individual
    return best_fitness_values, best_fitness, best_individual
#########################################################################
#Calculat ethe decimale value of the best individuals
#########################################################################
def best_individual_cal(individual):
    individual = individual[::-1]  # Reverse the binary string
    decimal_value = 0
    for i in range(len(individual)):
        decimal_value += individual[i] * 2 ** i  # Convert binary to decimal
    return decimal_value
#########################################################################
#Calculate the actual threshold value
########################################################################
def actual_threshold_value(individual):
    num=best_individual_cal(individual)
    threshold=scale_numbers(num, target_min=min_diff, target_max=max_diff)
    return threshold
#########################################################################
#########################################################################
#parameters of the genetic algorithm
population_size_input = int(input("Enter the population size: "))
chromosome_length_input =8
generations_input = int(input("Enter the generations: "))
crossover_rate_input = float(input("Enter the crossover rate: "))
mutation_rate_input = float(input("Enter the mutation rate: "))
consecutive_generations_threshold_input = int(input("Enter the consecutive generations threshold: "))

#########################################################################
# Calling the genetic algorithm function
best_fitness_values, best_fitness, best_individual = genetic_algorithm(population_size_input, chromosome_length_input, mutation_rate_input, crossover_rate_input, generations_input, consecutive_generations_threshold_input)
print("Best Minimum Fitness:", best_fitness)
print("Corresponding Individual:", best_individual)
best_threshold=actual_threshold_value(best_individual)
print("The best threshold value is : ",best_threshold)

##########################################################################
#ploting
##########################################################################
plt.figure(figsize=(7,7))
plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values, label='Min Fitness', color='b')
plt.title('Minimization Problem')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.legend()
plt.tight_layout()
plt.show()


