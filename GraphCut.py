import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy

# from google.colab.patches import cv2_imshow
from skimage import segmentation, io, color
import skimage
# from sklearn import preprocessing
import math
from skimage.color import rgb2gray
# import pytorch



def cluster_centers(superpixel_map):
    """
    This function takes a superpixel map and returns a list with the
    (row,col) positions of the cluster centers for that map
    """

    unique_labels = np.unique(superpixel_map)
    cluster_center_list = []

    for label_id, superpixel_label in enumerate(unique_labels):

        # Compute the coordinates where we have the superpixel_map = current label

        cluster_indices = np.where(superpixel_map == superpixel_label)

        # Compute the centroid of the coordinates to find the cluster centers

        centers = np.round(np.mean(cluster_indices, axis = 1)).astype('int')
        cluster_center_list.append(centers)

    return cluster_center_list

def cluster_centers_and_colors(superpixel_map, image):
    """
    This function takes a superpixel map and an image, and returns two lists:
    one with the (row, col) positions of the cluster centers, and another with
    the average color of each superpixel area.

    Parameters:
    superpixel_map (numpy.ndarray): The superpixel segmentation map.
    image (numpy.ndarray): The original image (same dimensions as superpixel_map).

    Returns:
    list of tuples: List of (row, col) tuples representing the cluster centers.
    list of tuples: List of (R, G, B) tuples representing the average color of each superpixel.
    """

    unique_labels = np.unique(superpixel_map)
    cluster_center_list = []
    average_color_list = []

    for label_id, superpixel_label in enumerate(unique_labels):
        # Get the indices for the current superpixel
        cluster_indices = np.where(superpixel_map == superpixel_label)

        # Compute the centroid of the coordinates to find the cluster centers

        centers = np.round(np.mean(cluster_indices, axis = 1)).astype('int')
        cluster_center_list.append(centers)

        # Extract the pixel values for the current superpixel
        pixels = image[cluster_indices]

        # Compute the average color of the superpixel
        average_color = np.mean(pixels, axis=0).astype('int')
        average_color_list.append(tuple(average_color))  # (R, G, B)

    return cluster_center_list, average_color_list

# Example usage:
# centers, average_colors = cluster_centers_and_colors(superpixel_map, image)


def apply_supermap(img, superpixel_map):
    """ This function returns an image where we assign the color of the cluster centers
    to every pixel of their corresponding segmentation groups."""
    centers = cluster_centers(superpixel_map)
    avg = []
    out = np.zeros_like(img)
    for i,(row, col) in enumerate(centers):
        out[superpixel_map == i] = img[row, col]
        # average_color = img[superpixel_map == i].mean(axis=0)
        # avg.append(average_color)

    # for i,(row, col) in enumerate(centers):
    #     out[superpixel_map == i] = avg[i]
    return out# , avg
# def apply_supermap(img, superpixel_map):
#     """
#     This function returns an image where we assign the average color of the
#     superpixel to every pixel of their corresponding segmentation groups.
#     """
#     out = np.zeros_like(img)
#     for superpixel_label in np.unique(superpixel_map):
#         mask = superpixel_map == superpixel_label
#         average_color = img[mask].mean(axis=0)  # Compute the average color for the current superpixel
#         out[mask] = average_color  # Assign the average color to all pixels in the superpixel
#     return out

def color_histogram(img, mask, num_bins):
    """For each channel in the image, compute a color histogram with the number of bins
    given by num_bins of the pixels in
    image where the mask is true. Then, concatenate the vectors together into one column vector (first
    channel at top).

    Mask is a matrix of booleans the same size as image.

    You MUST normalize the histogram of EACH CHANNEL so that it sums to 1.
    You CAN use the numpy.histogram function.
    You MAY loop over the channels.
    The output should be a 3*num_bins vector because we have a color image and
    you have a separate histogram per color channel.

    Hint: np.histogram(img[:,:,channel][mask], num_bins)"""

    rows, cols, channels = img.shape
    histogram = np.zeros(num_bins*3)

    # ===============================================
    dum = []
    count = 0
    for i in range(channels):
        cur = np.histogram(img[:,:,i][mask], bins=num_bins)[0]
        cur = cur / sum(cur)
        dum.append(cur)

    histogram = np.concatenate(dum, 0)
    # ===============================================
    return histogram

def adjacencyMatrix(superpixel_map):
    """Implement the code to compute the adjacency matrix for the superpixel map
    The input is a superpixel map and the output is a binary adjacency matrix NxN
    (N being the number of superpixels in svMap).  Bmap has a 1 in cell i,j if
    superpixel i and j are neighbors. Otherwise, it has a 0.  Superpixels are neighbors
    if any of their pixels are neighbors."""

    segmentList = np.unique(superpixel_map)
    segmentNum = len(segmentList)
    adjMatrix = np.zeros((segmentNum, segmentNum))

    # ===============================================
    for x in range(superpixel_map.shape[0]):
        for y in range(superpixel_map.shape[1]):
            if x + 1 < superpixel_map.shape[0] and superpixel_map[x][y] != superpixel_map[x + 1][y]:
                adjMatrix[superpixel_map[x][y]][superpixel_map[x + 1][y]] = 1
            if x - 1 >= 0 and superpixel_map[x][y] != superpixel_map[x - 1][y]:
                adjMatrix[superpixel_map[x][y]][superpixel_map[x - 1][y]] = 1
            if y + 1 < superpixel_map.shape[1] and superpixel_map[x][y] != superpixel_map[x][y + 1]:
                adjMatrix[superpixel_map[x][y]][superpixel_map[x][y + 1]] = 1
            if y - 1 >= 0 and superpixel_map[x][y] != superpixel_map[x][y - 1]:
                adjMatrix[superpixel_map[x][y]][superpixel_map[x][y - 1]] = 1

    # ===============================================

    return adjMatrix

def average_node_degree(adjMatrix):
    """ This function takes an adjacency matrix and returns
    the average number of neighborghs that the segments have
    (average node degree)"""

    # ===============================================
    # TODO: replace pass with your code
    average_node_degree = np.count_nonzero(adjMatrix) / len(adjMatrix)

    # ===============================================

    return average_node_degree

#This class represents a directed graph using adjacency matrix representation
class Graph:

    def __init__(self,graph):
        self.graph = graph # residual graph
        self. ROW = len(graph)
        # self.COL = len(gr[0])


    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''
    def BFS(self,s, t, parent):

        # Mark all the vertices as not visited
        visited =[False]*(self.ROW)

        # Create a queue for BFS
        queue=[]

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            #Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0 :
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False


    # Returns tne current flow from s to t in the given graph
    def FordFulkerson(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1]*(self.ROW)

        max_flow = 0 # There is no flow initially
        current_flow = np.zeros_like(self.graph)

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent) :
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while(s !=  source):
                path_flow = min (path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow +=  path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while(v !=  source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                current_flow[u][v] += path_flow
                current_flow[v][u] -= path_flow
                v = parent[v]

        return current_flow


def reduce(img, superpixel_map, num_bins=10):
    """This function takes as input an image, its corresponding superpixel map, and a
    number of bins as input. The output is a list of feature vectors.
    Each feature vector is the resulting histogram from applying the color_histogram
    function you implemented to every segment on the superpixel map."""

    feature_vectors = []
    num_segments = len(np.unique(superpixel_map))
    for i in range(num_segments):
        mask = superpixel_map == i
        feature_vectors.append(color_histogram(img, mask, num_bins))
    return(feature_vectors)

def graph_cut(superpixel_map, features, centers, keyindex):
    """Function to take a superpixel set and a keyindex and convert to a
    foreground/background segmentation.

    keyindex is the index to the superpixel segment we wish to use as foreground and
    find its relevant neighbors.

    centers is a list of tuples (row, col) with the positions of the cluster centers
    of the superpixel_map

    features is a list of histograms (obtained from the reduce function) for every superpixel
    segment in an image.

    """

    #Compute basic adjacency information of superpixels
    #Note that adjacencyMatrix is code you need to implement

    # ===============================================
    # TODO: this should be one line of code

    adjMatrix = adjacencyMatrix(superpixel_map)

    # ===============================================


    # normalization for distance calculation based on the image size
    # for points (x1,y1) and (x2,y2), distance is
    # exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
    dnorm = 2*(superpixel_map.shape[0]/2 *superpixel_map.shape[1] /2)**2
    k = len(features) #number of superpixels in image

    #Generate capacity matrix
    capacity = np.zeros((k+2,k+2))
    source = k
    sink = k+1

    # This is a single planar graph with an extra source and sink
    #  Capacity of a present edge in the graph is to be defined as the product of
    #  1:  the histogram similarity between the two color histogram feature vectors.
    #  The similarity between histograms should be computed as the intersections between
    #  the histograms. i.e: sum(min(histogram 1, histogram 2))
    #  2:  the spatial proximity between the two superpixels connected by the edge.
    #      use exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
    #
    #  Source gets connected to every node except sink
    #  Capacity is with respect to the keyindex superpixel
    #  Sink gets connected to every node except source and its capacity is opposite
    # The weight between a pixel and the sink is going to be the max of all the weights between
    # the source and the image pixels minus the weight between that specific pixel and the source.
    # Other superpixels get connected to each other based on computed adjacency
    # matrix: the capacity is defined as above, EXCEPT THAT YOU ALSO NEED TO MULTIPLY BY A SCALAR 0.25 for
    # adjacent superpixels.


    key_features = features[keyindex] # color histogram representation of superpixel # keyindex
    key_x = centers[keyindex][1] # row of cluster center for superpixel # keyindex
    key_y =  centers[keyindex][0] # col of cluster center for superpixel # keyindex
    # capacity is a K+2 x K+2 matrix
    # k is source and k+1 is sink
    color_weight = 50  # Increase this to give more weight to color similarity
    spatial_weight = 1  # Decrease this to reduce the influence of spatial proximity
    
    for i in range(k):
        for j in range(k):
            if adjMatrix[i, j] == 1:
                # 1 - Histogram similarity
                hist_sum = np.sum(np.minimum(features[i], features[j])) * color_weight
                # 2 - Spatial proximity
                prox = np.exp(-1 * ((centers[i][1] - centers[j][1]) ** 2 + (centers[i][0] - centers[j][0]) ** 2) / dnorm) *  spatial_weight
                capacity[i,j] = 0.25 * hist_sum * prox

    
    for i in range(k):
        # 1 - Histogram similarity
        # added the color intensity
        hist = np.sum(np.minimum(key_features, features[i])) * 2 *  color_weight
        # 2 - Spatial proximity
        prox = (np.exp(-1 * ((key_x - centers[i][1]) ** 2 + (key_y - centers[i][0]) ** 2) / dnorm)) * spatial_weight
        capacity[source, i] = hist * prox

    max_val = max(capacity[source])
    
    for i in range(k):
        capacity[i, sink] = max_val - capacity[source, i]


    # ===============================================

    # Obtaining the current flow of the graph when the flow is max
    g = Graph(capacity.copy())
    current_flow = g.FordFulkerson(source, sink)

    # Extract the two-class segmentation.
    # the cut will separate all nodes into those connected to the
    # source and those connected to the sink.
    # The current_flow matrix contains the necessary information about
    # the max-flow through the graph.

    segment_map = np.zeros_like(superpixel_map)
    rem_capacity = capacity - current_flow

    # ===============================================
    # TODO: Do the segmentation and fill segmentation map with 1s where the foreground is.
    # Replace pass with your code
    rows, cols = rem_capacity.shape
    for i in range(rows):
        for j in range(cols):
            if np.isclose(rem_capacity[i, j], 0):
                rem_capacity[i, j] = 0

    id_found = [source]
    search = [source]
    while len(search) != 0:
        curr = search[0]
        search = search[1:]
        for x in range(adjMatrix.shape[0]):
            if rem_capacity[curr, x] != 0 and (x not in id_found) and x != curr:
                id_found.append(x)
                search.append(x)
    id_found = np.unique(id_found)

    sup_rows, sup_cols = superpixel_map.shape

    for i in range(sup_rows):
        for j in range(sup_cols):
            if superpixel_map[i, j] in id_found:
                segment_map[i, j] = 1
            else:
                segment_map[i, j] = 0

    # ===============================================

    return capacity, segment_map

def find_nontree_superpixel(image, centers):
    """
    Finds the most green superpixel center in an image.

    Parameters:
    image (numpy.ndarray): The input image in RGB format.
    centers (list of tuples): List of (row, col) tuples representing superpixel centers.

    Returns:
    int: The index of the most green superpixel center.
    """

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])  # Adjust these values based on your needs
    upper_green = np.array([80, 255, 255])  # Adjust these values based on your needs

    max_green_value = -1
    most_green_index = -1

    # Iterate over each center and check its green value
    for i, (row, col) in enumerate(centers):
        # Extract the HSV values of the pixel at the center
        hsv_value = hsv_image[row, col]

        # Check if the hue value is within the green range
        if lower_green[0] <= hsv_value[0] <= upper_green[0]:
            green_value = hsv_value[1] * hsv_value[2]  # Consider saturation and brightness

            # Update the most green superpixel index
            if green_value > max_green_value:
                max_green_value = green_value
                most_green_index = i

    return most_green_index

def find_green_from_rgb(rgb_list):
    """
    Converts a list of RGB values to HSV and finds the index of the most green color.

    Parameters:
    rgb_list (list of tuples): List of (R, G, B) tuples.

    Returns:
    int: The index of the most green color.
    """

    # Convert the list to a numpy array and reshape for cv2.cvtColor
    rgb_array = np.array(rgb_list).reshape((len(rgb_list), 1, 3)).astype(np.uint8)

    # Convert RGB to HSV
    hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

    # Adjusted range for green color in HSV
    lower_green = np.array([35, 40, 40])  # lower bound of green hue
    upper_green = np.array([85, 255, 255])  # upper bound of green hue

    max_green_index = -1
    max_green_value = -1

    # Iterate over each HSV value and find the most green
    for i, hsv in enumerate(hsv_array):
        if lower_green[0] <= hsv[0][0] <= upper_green[0]:
            green_value = hsv[0][1] * hsv[0][2]  # Use saturation and value to determine greenness

            if green_value > max_green_value:
                max_green_value = green_value
                max_green_index = i

    return max_green_index





# Visible 6
# pic1 = plt.imread('./visible6/2015_05_26.png')
# pic1 = pic1[:,:,:3]


# porch = plt.imread('./visible6/2015_09_15.png')
# porch = porch[:,:,:3]


# flower = plt.imread('./visible6/2017_10_06.png')
# flower = flower[:,:,:3]

# Visible 10
# pic1 = plt.imread('./visible10/2014_03_06_cloud_0.png')
pic1 = plt.imread('./visible10/enhanced_1.jpg')
pic1 = pic1[:,:,:3]


porch = plt.imread('./visible10/2015_06_29_cloud_0.png')
porch = porch[:,:,:3]


flower = plt.imread('./visible10/2016_07_01_cloud_0.png')
flower = flower[:,:,:3]

# to increase contrast/saturation

#600
super_pic1 = segmentation.slic(pic1, n_segments=600, compactness=10, max_num_iter=20) - 1

super_porch = segmentation.slic(porch, n_segments=600, compactness=10, max_num_iter=20) - 1

super_flower = skimage.segmentation.slic(flower, n_segments=600, compactness=10, max_num_iter=20) - 1


fig, ax = plt.subplots(3, 4, figsize=(25,25))
images = [pic1, porch, flower]
maps = [super_pic1, super_porch, super_flower]

out_images = [apply_supermap(image, maps[i]) for i, image in enumerate(images)]
# out_images = []
# avg_colors = []
# for i, image in enumerate(images):
#     out1, avg1 = apply_supermap(image, maps[i])
#     out1 = apply_supermap
#     out_images.append(out1)
#     avg_colors.append(avg1)

for i,a in enumerate(ax):
    a[0].set_axis_off()
    a[0].set_title('Original', fontsize=20)
    a[0].imshow(images[i])
    
    a[1].set_axis_off()
    a[1].set_title('Superpixel map', fontsize=20)
    a[1].imshow(skimage.color.label2rgb(maps[i]))
    a[2].set_axis_off()
    a[2].set_title('Map overlayed on original', fontsize=20)
    a[2].imshow(skimage.color.label2rgb(maps[i], images[i]))
    a[3].set_axis_off()
    a[3].set_title('Color from cluster centers', fontsize=20)
    a[3].imshow(out_images[i])
    # a[3].imshow(apply_supermap(images[i], maps[i]))

fig.savefig("test.png")




pic1_features = reduce(pic1, super_pic1)
pic1_centers = cluster_centers(super_pic1)
# pic1_centers, avg_color = cluster_centers_and_colors(super_pic1, images[0])

porch_features = reduce(porch, super_porch)
porch_centers = cluster_centers(super_porch)

flower_features = reduce(flower, super_flower)
flower_centers = cluster_centers(super_flower)
most_green_pic11 = 0
val = float('inf')

for i,(row, col) in enumerate(pic1_centers):
    # if val < images[0][row, col, 1]:
    #     val = images[0][row, col, 1]
    #     most_green_pic11 = i
    if val > images[0][row, col, 1]:
        val = images[0][row, col, 1]
        most_green_pic11 = i

# most_green_pic11 = find_green_from_rgb(avg_colors[0])
# most_green_veggies2 = find_green_from_rgb(avg_colors[1])
# most_green_veggies3 = find_green_from_rgb(avg_colors[2])

# most_green_pic11 = find_nontree_superpixel(images[0], pic1_centers)
most_green_veggies2 = find_nontree_superpixel(images[1], porch_centers)
most_green_veggies3 = find_nontree_superpixel(images[2], flower_centers)

pic1_capacity, pic1_segment_map = graph_cut(super_pic1, pic1_features, pic1_centers, most_green_pic11)

porch_capacity,porch_segment_map = graph_cut(super_porch, porch_features, porch_centers, most_green_veggies2)

flower_capacity,flower_segment_map = graph_cut(super_flower, flower_features, flower_centers, most_green_veggies3)

fig, ax = plt.subplots(3, 3, figsize=(25,25))

ax[0][0].set_axis_off()
ax[0][0].set_title('March, 2014', fontsize=20)
ax[0][0].imshow(pic1, cmap="gray")
ax[0][1].set_axis_off()
ax[0][1].set_title('June, 2015', fontsize=20)
ax[0][1].imshow(porch, cmap="gray")
ax[0][2].set_axis_off()
ax[0][2].set_title('July, 2016', fontsize=20)
ax[0][2].imshow(flower, cmap="gray")
ax[0][2].set_axis_off()

ax[1][0].set_axis_off()
ax[1][0].set_title('Superpixel selected', fontsize=20)
ax[1][0].imshow(super_pic1==most_green_pic11, cmap="gray")
ax[1][1].set_axis_off()
ax[1][1].set_title('Superpixel selected', fontsize=20)
ax[1][1].imshow(super_porch==most_green_veggies2, cmap="gray")
ax[1][2].set_axis_off()
ax[1][2].set_title('Superpixel selected', fontsize=20)
ax[1][2].imshow(super_flower==most_green_veggies3, cmap="gray")
ax[1][2].set_axis_off()

ax[2][0].set_axis_off()
ax[2][0].set_title('2014: Segmentation of trees', fontsize=20)
ax[2][0].imshow(pic1_segment_map, cmap="gray")
ax[2][1].set_axis_off()
ax[2][1].set_title('2015: Segmentation of trees', fontsize=20)
ax[2][1].imshow(porch_segment_map, cmap="gray")
ax[2][2].set_axis_off()
ax[2][2].set_title('2016: Segmentation of trees', fontsize=20)
ax[2][2].imshow(flower_segment_map, cmap="gray")
ax[2][2].set_axis_off()

fig.savefig("test2.png")