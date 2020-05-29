import sys
import time
import numpy as np
from sklearn.cluster import KMeans
import math
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

start_time = time.time()
np.set_printoptions(threshold=sys.maxsize)
input_filename = sys.argv[1]
num_clusters = int(sys.argv[2])
output_filename = sys.argv[3]


input_file = open(input_filename,"r")
input_data = input_file.readlines()
input_file.close()

data = np.array(input_data)
np.random.shuffle(data)

Result = {}
n_DS_points = 0
n_CS_points = 0
n_RS_points = 0
n_CS_clusters = 0
round = 1
def update_CS_statistics(new_point,assigned_cluster):
    
    cluster = CS_summary[assigned_cluster]
    # POINTS
    cluster[0].append(list(new_point))

    # N
    cluster[1] = len(CS_summary[assigned_cluster][0])

    # SUM
    cluster[2] = np.add(cluster[2],new_point.astype(np.float))

    #SUMSQ
    cluster[3] = np.add(cluster[3],(new_point.astype(np.float))**2)

    # CENTROID
    cluster[4] = cluster[2] / cluster[1]

    # STANDARD DEVIATION
    variance = (cluster[3]/cluster[1]) - ((cluster[2]/cluster[1])**2)
    standard_deviation = np.sqrt(variance)
    cluster[5] = standard_deviation


def merge_cs_clusters(cluster1,cluster2):
    if (cluster1 in CS_summary and cluster2 in CS_summary):
        c1 = cluster1
        c2 = cluster2
        # points
        CS_summary[c1][0].extend(CS_summary[c2][0])
        # N
        CS_summary[c1][1] = len(CS_summary[c1][0])

        # SUM
        CS_summary[c1][2] = np.add(CS_summary[c1][2], CS_summary[c2][2])
        # SUMSQ
        CS_summary[c1][3] = np.add(CS_summary[c1][3], CS_summary[c2][3])
        # CENTROID
        CS_summary[c1][4] = CS_summary[c1][2] / CS_summary[c1][1]
        # STANDARD DEVIATION
        variance = (CS_summary[c1][3] / CS_summary[c1][1]) - ((CS_summary[c1][2] / CS_summary[c1][1]) ** 2)
        CS_summary[c1][5] = np.sqrt(variance)
        del CS_summary[c2]

def merge_cs_to_ds(ds_key,cs_key):
    if (ds_key in DS_summary and cs_key in CS_summary):

        # points
        DS_summary[ds_key][0].extend(CS_summary[cs_key][0])
        # N
        DS_summary[ds_key][1] = len(DS_summary[ds_key][0])
        # SUM
        DS_summary[ds_key][2] = np.add(DS_summary[ds_key][2], CS_summary[cs_key][2])
        # SUMSQ
        DS_summary[ds_key][3] = np.add(DS_summary[ds_key][3], CS_summary[cs_key][3])
        # CENTROID
        DS_summary[ds_key][4] = DS_summary[ds_key][2] / DS_summary[ds_key][1]
        # STANDARD DEVIATION
        variance = (DS_summary[ds_key][3] / DS_summary[ds_key][1]) - ((DS_summary[ds_key][2] / DS_summary[ds_key][1]) ** 2)
        DS_summary[ds_key][5] = np.sqrt(variance)
        del CS_summary[cs_key]

def get_CS_from_RS():
    global RS_points
    X = np.array(RS_points)
    large_k = 5 * num_clusters

    if len(RS_points) <= large_k:
        large_k = len(RS_points)

    kmeans4 = KMeans(n_clusters=large_k).fit(X)
    current_CS_cluster_indices = defaultdict(list)
    count = 0
    for cluster_id in kmeans4.labels_:
        current_CS_cluster_indices[cluster_id].append(RS_points[count])
        count += 1
    #print("Before Merge of RS to CS", len(CS_summary))

    for cluster_no, cluster_points in current_CS_cluster_indices.items():
        if len(cluster_points) > 1:
            # print("rs point has to be deleted")
            k = min(CS_summary.keys()) if len(CS_summary) != 0 else 0
            if cluster_no in CS_summary:
                while k in CS_summary:
                    k += 1
            else:
                k = cluster_no
            CS_summary[k] = defaultdict(dict)

            # points - not actual points but only indices store
            CS_summary[k][0] = list(cluster_points)
            # N
            CS_summary[k][1] = len(cluster_points)
            # SUM
            CS_summary[k][2] = np.sum(np.array(current_CS_cluster_indices[cluster_no]).astype(np.float), axis=0)
            # SUMSQ
            CS_summary[k][3] = np.sum((np.array(current_CS_cluster_indices[cluster_no]).astype(np.float)) ** 2, axis=0)
            # centroid
            CS_summary[k][4] = CS_summary[k][2] / CS_summary[k][1]

            # variance and standard  deviation
            variance = (CS_summary[k][3] / CS_summary[k][1]) - ((CS_summary[k][2] / CS_summary[k][1]) ** 2)
            standard_deviation = np.sqrt(variance)
            CS_summary[k][5] = standard_deviation

    #print("After Merge of RS to CS", len(CS_summary))
    del_set = []
    #print("Before merging RS Length is", len(RS_points))

    for k, v in current_CS_cluster_indices.items():
        if len(v) > 1:
            for point in v:
                del_set.append(RS_points.index(point))

    RS_points = np.delete(RS_points, del_set, axis=0).tolist()
    #print("After merging RS length is", len(RS_points))

# ********Step 1: load 20 % data randomly ****************

rand_length = int(len(data)*0.2)
left_over = len(data) - rand_length * 5

first_sample = data[:rand_length]
initial_points = []
outliers = 0
point_to_index = {}
ground_truth = {}

for idx,row in enumerate(first_sample):
    new_row = row.replace("\n","").split(",")
    #if new_row[1] == '-1':
       #   outliers += 1
    point_to_index[tuple(new_row[2:])] = new_row[0]
    #ground_truth[int(new_row[0])] = new_row[1]
    initial_points.append(new_row[2:])

dimensions = len(initial_points[0])
mahalanobis_threshold = 2 * math.sqrt(dimensions)


#********Step 2: Run K-means *****************************

X = np.array(initial_points)
kmeans = KMeans(n_clusters=5*num_clusters).fit(X)

initial_clusters_indices = defaultdict(list)
count = 0
for cluster_id in kmeans.labels_:
    initial_clusters_indices[cluster_id].append(initial_points[count])
    count += 1


#********Step 3: Move outliers to RS ***********************

RS_points = []
for k,v in initial_clusters_indices.items():
    if len(v) == 1:
        RS_points.append(list(v[0]))
        initial_points.remove(v[0])


#********Step 4: Run K- means again *******************

X = np.array(initial_points)
kmeans2 = KMeans(n_clusters=num_clusters).fit(X)

current_clusters_indices = defaultdict(list)
count = 0
for cluster_id in kmeans2.labels_:
    current_clusters_indices[cluster_id].append(initial_points[count])
    count += 1

#**************** Step 5: Update Discard set statistics ******************

DS_summary = {}
for cluster_no,cluster_points in current_clusters_indices.items():
    DS_summary[cluster_no] = defaultdict(dict)
    #points
    DS_summary[cluster_no][0] = cluster_points
    #N
    DS_summary[cluster_no][1] = len(DS_summary[cluster_no][0])
    #SUM
    DS_summary[cluster_no][2] = np.sum(np.array(current_clusters_indices[cluster_no]).astype(np.float), axis=0)
    #SUMSQ
    DS_summary[cluster_no][3] = np.sum((np.array(current_clusters_indices[cluster_no]).astype(np.float)) ** 2, axis=0)
    #centroid
    centroid = DS_summary[cluster_no][2] / DS_summary[cluster_no][1]
    DS_summary[cluster_no][4] = centroid
    #variance and standard  deviation
    variance = (DS_summary[cluster_no][3] / DS_summary[cluster_no][1]) - ((DS_summary[cluster_no][2] / DS_summary[cluster_no][1]) ** 2)
    standard_deviation = np.sqrt(variance)
    DS_summary[cluster_no][5] = standard_deviation

# ***************** Step 6: Create CS from points in RS ************************

#print("No.of RS points",len(RS_points))
large_k = 5 * num_clusters

if len(RS_points) <= large_k:
    large_k = len(RS_points)

X = np.array(RS_points)
kmeans3 = KMeans(n_clusters=large_k).fit(X)

CS_clusters_indices = defaultdict(list)
count = 0
for cluster_id in kmeans3.labels_:
    CS_clusters_indices[cluster_id].append(RS_points[count])
    count += 1

CS_summary = {}
for cluster_no, cluster_points in CS_clusters_indices.items():
    if len(cluster_points) > 1:
        CS_summary[cluster_no] = defaultdict(dict)

        #points
        CS_summary[cluster_no][0]= list(cluster_points)
        #N
        CS_summary[cluster_no][1] = len(cluster_points)
        #SUM
        CS_summary[cluster_no][2] = np.sum(np.array(CS_clusters_indices[cluster_no]).astype(np.float), axis = 0)
        #SUMSQ
        CS_summary[cluster_no][3] = np.sum((np.array(CS_clusters_indices[cluster_no]).astype(np.float)) ** 2, axis=0)
        # centroid
        centroid = CS_summary[cluster_no][2] / CS_summary[cluster_no][1]
        CS_summary[cluster_no][4] = centroid
        # variance and standard  deviation
        variance = (CS_summary[cluster_no][3] / CS_summary[cluster_no][1]) - ((CS_summary[cluster_no][2] / CS_summary[cluster_no][1]) ** 2)
        standard_deviation = np.sqrt(variance)
        CS_summary[cluster_no][5] = standard_deviation

#print("Initial points in CS", len(CS_summary))
del_set = []
#print("Before merging to CS", len(RS_points))
for k, v in CS_clusters_indices.items():
   if len(v) > 1:
        #print("MERGE HAPPENED")
        for idx in v:
            del_set.append(RS_points.index(idx))
X = np.array(RS_points)

RS_points = np.delete(X,del_set,axis=0).tolist()

#print("After merging to CS",len(RS_points))
#summary
n_CS_clusters = len(CS_summary)
for key in CS_summary.keys():
    n_CS_points += CS_summary[key][1]
for key in DS_summary.keys():
    n_DS_points += DS_summary[key][1]
n_RS_points = len(RS_points)
#print("Round 1: "+str(n_DS_points)+","+str(n_CS_clusters)+","+str(n_CS_points)+","+str(n_RS_points))
Result[round] = [n_DS_points,n_CS_clusters,n_CS_points,n_RS_points]
round = 2
start = rand_length
end = start + rand_length
while round <= 5:
    if round == 5:
        end = end + left_over
        new_sample = data[start:end]
    else:
        new_sample = data[start:end]
        start = end
        end = start + rand_length

    new_points = []
    #*********Step 7: Load 20% of data randomly **************
    for idx, row in enumerate(new_sample):
        new_row = row.replace("\n", "").split(",")
        new_points.append(new_row[2:])
        #if new_row[1] == '-1':
         #   outliers += 1
        point_to_index[tuple(new_row[2:])] = new_row[0]
        #ground_truth[int(new_row[0])] = new_row[1]
    X = np.array(new_points)
    #*********Step 8: Assign new point to DS *********************
    unassigned_set = []
    for new_point in X:
        point = new_point.astype(np.float)
        min_dist = mahalanobis_threshold
        assigned_cluster = float("-inf")
        for cluster_no, value in DS_summary.items():
            centroid = DS_summary[cluster_no][4]
            standard_deviation = DS_summary[cluster_no][5]
            MD = np.sqrt(np.sum(np.square((point-centroid)/standard_deviation)))
            if MD < min_dist:
                min_dist = MD
                assigned_cluster = cluster_no
        if assigned_cluster > float("-inf"):
            cluster_ds = DS_summary[assigned_cluster]
            #POINTS
            cluster_ds[0].append(list(new_point))

            # N
            cluster_ds[1] = cluster_ds[1] + 1

            # SUM and SUMSQ
            cluster_ds[2] = np.add(cluster_ds[2],new_point.astype(np.float))
            cluster_ds[3] = np.add(cluster_ds[3],(new_point.astype(np.float)) ** 2)

            # CENTROID
            cluster_ds[4] = cluster_ds[2] / cluster_ds[1]

            # STANDARD DEVIATION
            variance = (cluster_ds[3] / cluster_ds[1]) - ((cluster_ds[2] / cluster_ds[1]) ** 2)
            standard_deviation = np.sqrt(variance)
            cluster_ds[5] = standard_deviation

        else: # *********** Step 9 : Try to assign to CS **********
            unassigned_set.append(new_point)
    for new_point in unassigned_set:
        point = new_point.astype(np.float)
        c_min_dist = mahalanobis_threshold
        assigned_cluster = float("-inf")
        for cluster_no, value in CS_summary.items():
            centroid = CS_summary[cluster_no][4]
            standard_deviation = CS_summary[cluster_no][5]
            MD = np.sqrt(np.sum(np.square((point - centroid) / standard_deviation)))
            if MD < c_min_dist:
                c_min_dist = MD
                assigned_cluster = cluster_no
        if assigned_cluster > float("-inf"):
            update_CS_statistics(new_point,assigned_cluster)
        else:
        # ************* Step 10: If not assign to RS set *******
            RS_points.append(list(new_point))

    # *************Step 11: Run Kmeans on RS to generate CS and RS************
    get_CS_from_RS()
    # *************Step 12: Merge CS clusters with less than threshold *******************
    #print("Before Merging CS clusters", len(CS_summary))

    for cluster1 in CS_summary.copy().keys():
        for cluster2 in CS_summary.copy().keys():
            if (cluster1 != cluster2 and cluster1 in CS_summary):

                standard_deviation1 = CS_summary[cluster1][5]
                standard_deviation2 = CS_summary[cluster2][5]

                centroid1 = CS_summary[cluster1][4]
                centroid2 = CS_summary[cluster2][4]

                # MD1 = np.sqrt(np.sum(np.square((centroid1-centroid2)/standard_deviation1)))
                # MD2 = np.sqrt(np.sum(np.square((centroid1 - centroid2)/standard_deviation2)))
                # MD = min(MD1,MD2)
                MD = np.sqrt((np.sum(np.square((centroid1-centroid2)/(standard_deviation2*standard_deviation1)))))
                if MD < mahalanobis_threshold:
                    merge_cs_clusters(cluster1,cluster2)

    #print("After merging CS clusters", len(CS_summary))

    if (round == 5):
        #print("round 5 reached")
        for cs_cluster in CS_summary.copy().keys():
            for ds_cluster in DS_summary.copy().keys():
                if (cs_cluster in CS_summary and ds_cluster in DS_summary):
                    standard_deviation1 = CS_summary[cs_cluster][5]
                    standard_deviation2 = DS_summary[ds_cluster][5]

                    centroid1 = CS_summary[cs_cluster][4]
                    centroid2 = DS_summary[ds_cluster][4]

                    # MD1 = np.sqrt(np.sum(np.square((centroid1 - centroid2) / standard_deviation1)))
                    # MD2 = np.sqrt(np.sum(np.square((centroid1 - centroid2) / standard_deviation2)))
                    # MD = min(MD1,MD2)
                    MD = np.sqrt((np.sum(np.square((centroid1 - centroid2) / (standard_deviation2 * standard_deviation1)))))
                    if MD < mahalanobis_threshold:
                        #print("FINALLLY MERGEDD !!!!!")
                        merge_cs_to_ds(ds_cluster,cs_cluster)

    n_DS_points = 0
    n_CS_points = 0
    n_RS_points = 0
    n_CS_clusters = 0

    for keys in DS_summary.keys():
        n_DS_points += DS_summary[keys][1]

    n_CS_clusters = len(CS_summary)

    for keys in CS_summary.keys():
        n_CS_points += CS_summary[keys][1]

    n_RS_points = len(RS_points)

    Result[round] = [n_DS_points,n_CS_clusters,n_CS_points,n_RS_points]
    #print("Round"+str(round)+":"+str(n_DS_points)+","+str(n_CS_clusters)+","+str(n_CS_points)+","+str(n_RS_points))
    round += 1

#print("Total outliers",outliers)
clustered_output = {}

for cluster in DS_summary:
    for points in DS_summary[cluster][0]:
        index = point_to_index[tuple(points)]
        clustered_output[int(index)] = cluster
for key,value in point_to_index.items():
    if int(value) not in clustered_output:
        clustered_output[int(value)] = -1
sorted_clustered_output = sorted(clustered_output)

predicted_values = []
for key in sorted_clustered_output:
    #print(key)
    predicted_values.append(clustered_output[key])

f = open(output_filename,"w")
f.write("The intermediate results:")
f.write("\n")
f.write("\n".join('Round {}: {},{},{},{}'.format(k,v[0],v[1],v[2],v[3]) for k,v in Result.items()))
f.write("\n\n\n")
f.write("The clustering results:")
f.write("\n")
for key in sorted_clustered_output:
    f.write(str(key)+","+str(predicted_values[key]))
    f.write("\n")
f.close()

#ground_truth_list = []

#for key in sorted(ground_truth):
#    ground_truth_list.append(ground_truth[key])

#X = np.array(predicted_values)
#Y = np.array(ground_truth_list)
#score = normalized_mutual_info_score(Y,X)

#print(len(clustered_output))
#print("Score",score)
print("Execution time",time.time() - start_time)

