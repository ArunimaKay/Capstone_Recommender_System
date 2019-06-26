#!/usr/bin/env python
# coding: utf-8

# # spaCy/HDBScan Feature Extraction Pipeline
# 
# ### Note: it can be quite complicated to install spaCy and sense2vec, given conflicting low-level requirementso, so at this point I wouldn't suggest that others try to install the libraries and run this notebook.
#   
# However, it is well worth scanning down to the cell titled ***Harvesting Word Features***. In the output of that cell, there are examples of 52 feature clusters harvested by this process. The ultimate output of this process will produce a dataset containing a product ID (asin), overall rating, and word feature, for each word feature found in each product review. I don't consider these feature clusters as the final product, and we should discuss.
# 
# 
# ### We can use this output for several purposes. 
# 
# 1. First, we should be able to quite easily make the data available to th web interface, so that we can display the top n word features (by overall rating) associated with products returned.
# 
# 2. We will want to also include the user's selected word features in our model evaluation, to enable them to "drill into" selected features and thus explore the product/feature landscape.
# 
# 3. Finally, I think it would be worth training a model on a vectorized representation of the top n most highly rated features, which may give us another dimension for predicting rating based on feature combination/interaction.

# In[1]:


#!conda uninstall -y spacy

#Installing spaCy :

# For Linux:
# !conda install -y spacy -c conda-forge
# (use "spacy[cuda100]", if you have the 10.0 cuda driver installed) 

# For Mac:
# !pip install spacy==2.0.7

# Installing HDBScan and sense2vec
# !conda install -y -c conda-forge hdbscan
# !pip install sense2vec==1.1.1a0


# In[2]:


#!pip install matplotlib
#!python -m pip list


# In[2]:


import pandas as pd
import gzip
import time
# Install a few python packages using pip
from common import utils


# In[1]:


# Standard python helper libraries.
import os, sys, time
import collections
import itertools

# Numerical manipulation libraries.
import numpy as np

#Visualization
import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')

import spacy
#activated = spacy.prefer_gpu()


import hdbscan
import seaborn as sns
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.tokens import Doc



#from sense2vec import Sense2VecComponent


plotting = False
labels_words = False

language_model = 'en_core_web_md'


start = time.time()
print("Reading English core web medium language data using spaCy...")
nlp = spacy.load(language_model)
print("...finished reading English language model '{}' in {} seconds.".format(language_model, time.time()-start))

#nlp_plain = spacy.load('en_core_web_md')
print("spaCy loaded en_core_web_md")
#s2v = Sense2VecComponent('/home/burgew/w210_data/reddit_vectors-1.1.0')

print("nlp: {}".format(nlp))
last_nlp_component = nlp.pipeline[-1]
#if last_nlp_component[0] != 'sense2vec':
#    nlp.add_pipe(s2v)
#    print("added sense2vec to spaCy NLP pipeline")
#else:
#    print("sense2vec previously added to spaCy NLP pipeline")


# In[9]:


print("nlp.pipeline[{}]: {}".format(len(nlp.pipeline), nlp.pipeline))


# In[10]:



debug = False

lemmas = {}
ignore_words = []


path_for_tf_metadata = './logdir/embedding_test'
path_for_tf_ckpt = path_for_tf_metadata+'/embedding_test.ckpt'
vectors_filepath = './data/vectors_each.tsv'
metadata_filepath = './data/metadata_each.tsv'


# In[12]:

import tqdm

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

vdim = file_len(vectors_filepath)
print("vectors file contains {} lines".format(vdim))
index = []
output = None

sample_prob=0.01
# Generate random samples set, used for both the vectors file and the metadata file so the elements match
samples = np.random.choice(a=[True,False], size=vdim, p=[sample_prob,1.0-sample_prob])
print(samples)



with open(vectors_filepath, 'r') as in_v:
    print('File {} contains sense vectors of dimension {}'.format(vectors_filepath, vdim))
    curr_line = 0
    t = tqdm.tqdm(total=vdim)
    
    for line in in_v:
        sample_this_row = samples[curr_line]
        
        if sample_this_row:
            if output is None:
                # Create an np.array with the first row as the retrieved word vector
                output = np.array([np.array(line.split('\t'))])
            else:
                # Append the next vector to the end of the vectors array
                output = np.append(output, np.array([np.array(line.split('\t'))]), axis=0)
                
        curr_line += 1
        t.update()
        #if len(output)>2:        
        #    break
    t.close()


# In[ ]:


print("Output shape: {}".format(output.shape))


# In[ ]:


print("Output{}: {}".format(output.shape, output[:5]))


HDBSCAN_METRIC = 'manhattan'

start = time.time()
print("Creating word clusters from word vectors...")
hdbscanner = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric=HDBSCAN_METRIC, gen_min_span_tree=True)
hdbscanner.fit(output)
print("...completed clustering in {} seconds.".format(time.time()-start))


import pickle 
with open('./data/hdbscanner.{}.pickle'.format(int(time.time())), 'wb') as pickle_file:
    pickle.dump(hdbscanner, pickle_file)


# In[ ]:

if plotting:
	start = time.time()
	print("Condensing the linkage tree and then plotting...")
	#hdbscanner.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
	hdbscanner.condensed_tree_.plot()
	hdbscanner.condensed_tree_.plot(select_clusters=True)
	print("...plotted condensed tree in {} seconds.".format(time.time()-start))
	tree = hdbscanner.condensed_tree_
	print("Found {} clusters".format(len(tree._select_clusters())))
	matplotlib.pyplot.show()


# 
# ### This can only be graphed as single linkage tree for very small datasets 
# 
# start = time.time()
# print("Plotting single linkage tree (not for large data) ...")
# hdbscanner.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
# print("...plotted single linkage tree tree in {} seconds.".format(time.time()-start))

# start = time.time()
# print("Plotting condensed tree...")
# hdbscanner.condensed_tree_.plot()
# print("...plotted condensed tree in {} seconds.".format(time.time()-start))

# In[ ]:

if plotting:
	start = time.time()
	print("Plotting condensed tree with vectors selected...")
	hdbscanner.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
	print("...plotted condensed tree with selected vectors in {} seconds.".format(time.time()-start))
	matplotlib.pyplot.show()


# In[ ]:



# In[ ]:


def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) & 
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)


# # Harvesting Word Features
# The below logic collects and filters word features from the condensed tree created in the cells above.

# In[ ]:


tree = hdbscanner.condensed_tree_

#print('Index, for reference:')
#for ind, entry in enumerate(index):
#    print("cluster: {}, ind: {}, entry: {}".format(hdbscanner.labels_[ind], ind, entry))

start = time.time()
print("Selecting clusters in tree...")
clusters = tree._select_clusters()
print("...finished selecting clusters in {} seconds.".format(time.time()-start))

initial_cluster_count = len(clusters)
print("Found {} clusters".format(initial_cluster_count))

all_points = []
labels = []

# iterate through the input metadata once, to collect all words and the labels for the sampled points
with open(metadata_filepath, 'r') as in_m, open(path_for_tf_metadata+'/metadata.tsv', 'w') as out_tf_meta:
	mdim = None
	curr_line = 0
	for line in in_m:
		if mdim is None:
			mdim = line.count('\t')+1
			print('File {} contains index entries of of dimension {}'.format(metadata_filepath, vdim))
		if line.endswith('\n'):
			line = line[:-1]
		all_points.append(line.split('\t'))
		sample_this_row = samples[curr_line]
		if sample_this_row:
			meta_line = line.split('\t')
			index.append(meta_line)
			if labels_words:
				labels.append(meta_line[2])
		curr_line += 1

print("Index({}, {}): {}".format(len(index), len(index[0]), index[:5]))

print("All Points({}, {}): {}".format(len(all_points), len(all_points[0]), all_points[:5]))

# then, iterate through the input metadata again, to apply the cluster labels, if labels_words is False
with open(metadata_filepath, 'r') as in_m, open(path_for_tf_metadata+'/metadata.tsv', 'w') as out_tf_meta:
	mdim = None
	curr_line = 0
	curr_sample = 0

	for line in in_m:
		if mdim is None:
			mdim = line.count('\t')+1
			print('File {} contains index entries of of dimension {}'.format(metadata_filepath, vdim))
		if line.endswith('\n'):
			line = line[:-1]
		sample_this_row = samples[curr_line]
		if sample_this_row:
			index.append(line.split('\t'))
			if not labels_words:
				labels.append("-")
			curr_sample += 1
		curr_line += 1

# In[33]:
selected_clusters = []

for i, c in enumerate(clusters):
	c_exemplars = exemplars(c, tree)
    
	#plt.scatter(data.T[0][c_exemplars], data.T[1][c_exemplars], c=palette[i], **plot_kwds)
    
	#print("Index: ", enumerate(index))
	#print("Output: ", output[:5])

	point_label = None
	cluster_exemplars = set()
	for ind, ex_ind in enumerate(c_exemplars):
		#print("Exemplar -- {} : {}".format(index[ex_ind][0], index[ex_ind][2]))
		cluster_exemplars.add(index[ex_ind][2])
		if point_label is None:
			point_label = index[ex_ind][2]
    
	members = set()
	for label_ind, label in np.ndenumerate(hdbscanner.labels_):
		if label == i:
			members.add(index[label_ind[0]][2])
			if not labels_words:
				labels[label_ind[0]] = point_label
            
            #print("Member: {} : {}".format(index[label_ind[0]][0], index[label_ind[0]][2]))
    
	exemplars_len = float(len(cluster_exemplars))
	members_len = float(len(members))
    
	if ((exemplars_len>0) and (len(members)>(2.0*exemplars_len))):
		#print("\nCluster {} persistence: {}".format(i, hdbscanner.cluster_persistence_.item(i)))
		#print("Cluster {} Exemplars: ".format(i),c_exemplars)
		#print("Cluster {} Exemplar Probabilities: ".format(i),[hdbscanner.probabilities_[ind] for ind in c_exemplars])
    
		example_cluster_exemplars = ", ".join(cluster_exemplars)
		example_cluster_members = ", ".join(members)
        
		selected_clusters.append([example_cluster_exemplars, example_cluster_members])

selected_cluster_count = len(selected_clusters)
if (selected_cluster_count>0):
    print("\nFound {} clusters ({}% of initially collected):".
          format(len(selected_clusters), 100.0*float(selected_cluster_count)/float(initial_cluster_count)))
    for example in selected_clusters:
        print("\nExemplars: {}".format(example[0]))
        print("Members: {}".format(example[1]))

with open(path_for_tf_metadata+'/metadata.tsv', 'w') as out_tf_meta:
	for label in labels:
		out_tf_meta.write(str(label)+'\n')
                                                                    
noise_count = sum([1 for label in hdbscanner.labels_ if label == -1])
print("\nThere were {} words that were considered noise.".format(noise_count))

                                  
#print("\nOutliers.")
#for label_ind, label in np.ndenumerate(hdbscanner.labels_):
#    if label == -1:
#        print("{} : {}".format(index[label_ind[0]][0], index[label_ind[0]][2]))


# In[ ]:


hdbscanner.labels_


# In[ ]:


import tensorflow as tf

# value = np.array(value)
# value = value.reshape([2, 4])
output_init = tf.constant_initializer(output)

print('fitting shape:')
tf.reset_default_graph()
with tf.Session() :
    embedding_var = tf.get_variable('embedding_var', shape=[len(output), len(output[0])], initializer=tf.constant_initializer(output), dtype=tf.float32)
    embedding_var.initializer.run()
    print(embedding_var.eval())
    
sess = tf.Session()

sess.run(embedding_var.initializer)


with open(path_for_tf_ckpt,'w') as f:
    f.write("Index\tLabel\n")
    for ind,label_line in enumerate(index):
        label = '{}:{}:{}'.format(label_line[0], label_line[1], label_line[2])
        f.write("%d\t%s\n" % (ind,label))


# In[ ]:


from tensorflow.contrib.tensorboard.plugins import projector

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Create summary writer.
    writer = tf.summary.FileWriter(path_for_tf_metadata, sess.graph)
    # Initialize embedding_var
    sess.run(embedding_var.initializer)
    # Create Projector config
    config = projector.ProjectorConfig()
    # Add embedding visualizer
    embedding = config.embeddings.add()
    # Attache the name 'embedding'
    embedding.tensor_name = embedding_var.name
    # Metafile which is described later
    embedding.metadata_path = 'metadata.tsv'
    # Add writer and config to Projector
    projector.visualize_embeddings(writer, config)
    # Save the model
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, path_for_tf_ckpt, 1)

writer.close()


# In[ ]:




