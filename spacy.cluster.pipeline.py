#! /home/ubuntu/miniconda3/envs/py3.5/bin/python3.5 -u


# coding: utf-8

# In[ ]:





# # spaCy/HDBScan Feature Extraction Pipeline
# 
# ### Note: it can be quite complicated to install spaCy and sense2vec, given conflicting low-level requirements, so at this point I wouldn't suggest that others try to install the libraries and run this notebook.
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

from __future__ import print_function
from __future__ import division

import argparse
import pandas as pd
import gzip
import random
import time
# Install a few python packages using pip

#!conda install pytables

#!pip install PyEnchant
import enchant
#help(enchant)

english_dict = enchant.Dict("en_US")


# PyTables will be used to create huge matrices stored on disk, via HDF
import tables
from tables import *

from common import utils
utils.require_package("wget")      # for fetching dataset


# In[2]:


# Standard python helper libraries.
import os, sys, time
import collections
from collections import Counter
import itertools

# Numerical manipulation libraries.
import numpy as np

import pickle 

#Visualization
import matplotlib
#%matplotlib inline

import spacy
#activated = spacy.prefer_gpu()

import hdbscan
import html
import seaborn as sns
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.tokens import Doc
import tqdm
utils.require_package("wget") 
import nltk
from nltk.corpus import stopwords


# In[3]:


#%lsmagic

non_clustered_product_features = False
review_vectors = True
clustering = True
plotting = False
labeling = True
labels_words = False
text_features = True
tensor_vis = False

parser = argparse.ArgumentParser('Generate product review features for a specified build_set.')
parser.add_argument('build_set', help='the integer used to label products in a grouping of product categories')

args = parser.parse_args()
build_set = args.build_set
#build_set = 2

language_model = 'en_core_web_md'

start = time.time()
print("Reading spaCy language model {}...".format(language_model))
nlp = spacy.load(language_model, entity=False)
print("...finished reading English language model '{}' in {} seconds.".format(language_model, time.time()-start))

from nltk.corpus import stopwords
 
stopWords = set(stopwords.words('english'))
for stop_word in stopWords:
    nlp.vocab[stop_word].is_stop = True


# In[4]:


debug = False

# Paths for output files
path_for_tf_metadata = './logdir/embedding_test'
path_for_tf_ckpt = path_for_tf_metadata+'/embedding_test.ckpt'


# In[5]:


def display_local_time():
    localTime = time.localtime()
    print("Local time = {}:{}".format(localTime.tm_hour, format(localTime.tm_min,'02')))    


# In[6]:


def parse(path):
  print('start parse')
  start_parse = time.time()
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
  end_parse = time.time()
  print('end parse with time for parse',end_parse - start_parse)

def getDF(path):
  print('start getDF')
  start = time.time()
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  print('end getDF')
  end = time.time()
  print('time taken to load data = ',end-start)
  return pd.DataFrame.from_dict(df, orient='index')

start = time.time()
print("Reading Pandas dataframe from reviews_Toys_and_Games.json.gz...")
df = getDF('./data/reviews_Toys_and_Games_5.json.gz')
print("...read reviews_Toys_and_Games_5.json.gz (length={}) in {} seconds.".format(len(df), time.time()-start))


# In[8]:


# meta_Toys_and_Games.json.gz
print("Loading related metadata dataset:")
print()
# file = 'metadata.json.gz'
#file = './data/meta_Toys_and_Games.json.gz'
file = './data/meta_sorted_combined_Toys.json.gz'
start = time.time()
md = getDF(file)
print('Total time taken for loading the metadata dataset: {} minutes.'.format(np.round((time.time() - start)/60),2))
print()
print(md.columns)
print()
print("Number of records in metadata dataset: {}".format(len(md)))
print()
print(md.head(1))
print("\nNull values in metadata columns:")
md.isnull().sum()


# In[34]:


test_str = "Plan Toy Oval Xylophone is colorful and is an enjoyable way to stimulate children's natural sense of harmony and rhythm. This solid wood xylophone plays melodic, pleasant notes. Children will experiment with rhythm and note combinations to create their own songs. A wonderful introduction to music from any preschooler"
doc = nlp(test_str)
for sent in doc.sents:
    for chunk in sent.noun_chunks:
        print("**** Noun chunk: {} ****".format(chunk.text))
        for token in chunk:
            print("text:{}, lemma_:{}, pos_:{}, tag_:{}, dep_:{}, shape_:{}, is_alpha:{}, is_stop:{}, norm_:{}".format(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop, token.norm_))
            if not english_dict.check(token.norm_):
                print("Bad word! Suggestions:", english_dict.suggest(token.norm_))
        
        
        


# In[10]:


print("Count of rows where title and description are both null, out of {} total:".format(len(md)))
md[pd.isnull(md['title']) & pd.isnull(md['description'])].count()
print("This would indicate that there are no rows with both title and description being null.")

print()
print("Checking for missing values in the columns\n'['reviewerID', 'asin', 'reviewText', 'overall','imUrl', 'categories', 'related', 'price']'' of the combined dataset:")

dframe__ = md[['title', 'description']]

dframe_ = dframe__.dropna(axis = 0, how='all',inplace=False)

print()
print("There are {} records with missing values in both columns - \n['title', 'description']".format(len(dframe__) - len(dframe_)))

dframe_.head()


# In[11]:


print(df.shape)
print(df.columns)
df.head(2)


# In[12]:


#Number of reviews by product
cnt_by_product = df[['asin','reviewText']].groupby('asin').agg('count')
cnt_by_product['reviewText'].hist(bins = [0,1,2,3,4,5,10,15,20,50,100,500,1000])

print("summary of cnt_by_product:")
print(cnt_by_product[cnt_by_product['reviewText']>4][:20])
cnt_by_product[cnt_by_product['reviewText']>4].count()


# In[13]:


#Number of reviews by reviewer
cnt_by_reviewer = df[['reviewerID','reviewText']].groupby('reviewerID').agg('count').reset_index()

good_reviewers = cnt_by_reviewer[cnt_by_reviewer['reviewText']>4]
print("good_reviewers({}):\n".format(len(good_reviewers)), good_reviewers[:5])

products_by_review = len(df.groupby('asin')['reviewText'].count()>4)
print("\n# of products with greater than 4 reviews: ", products_by_review)

good_products_by_review = cnt_by_product[cnt_by_product['reviewText']>4].reset_index()
print("\ngood_products_by_reviews({}):\n".format(len(good_products_by_review)), good_products_by_review[:5])


# In[14]:



product_buildset = pd.read_csv('./data/cluster_build_sets.union.csv', names=['asin','build_set'])
print("\nproduct_buildset({}):\n".format(len(product_buildset)), product_buildset[:5])

build_set_filter = product_buildset['build_set']==int(build_set)
print("products with build_set=={}: {}".format(build_set, sum(build_set_filter)))

products_in_buildset = product_buildset[build_set_filter]
print("\nproducts_in_buildset({}):\n".format(len(products_in_buildset)), products_in_buildset[:5])

filters_merged = pd.merge(good_products_by_review, products_in_buildset[['asin','build_set']], on='asin')

print("\nfilters_merged({}):\n".format(len(filters_merged)), filters_merged[:5])

# Create a merged DF, joining the reviews data with the selected products with 5+ reviews in the a build set
df_merged = pd.merge(df[['asin','overall','reviewText']], filters_merged[['asin','build_set']], on='asin')
# Preprocess the reviews by HTML unescaping them
df_merged = df_merged.groupby(['asin','overall'], as_index=True)['reviewText'].apply(' '.join).apply(html.unescape).reset_index()
print("\n\ncleaned df columns:", list(df))
print("df_merged({}):\n".format(len(df_merged)), df_merged[:5])

# Create a merged DF, with asin, reviewerID and the overall rating per review
df_mean_merged = pd.merge(df[['asin','overall','reviewerID']], filters_merged[['asin']], on='asin').reset_index()
print("\n\ndf_mean_merged({}):\n".format(len(df_mean_merged)), df_mean_merged[:5])
# Aggregate the overall rating per revierwer
df_mean_merged = df_mean_merged.groupby(['asin','reviewerID'])['overall'].mean().reset_index()
print("\ndf_mean_merged({}):\n".format(len(df_mean_merged)), df_mean_merged[:5])
# Aggregate the per-reviewer mean rating for an overall per-product rating
df_mean_merged = df_mean_merged.groupby(['asin'])['overall'].mean().reset_index()
print("\ndf_mean_merged({}):\n".format(len(df_mean_merged)), df_mean_merged[:5])

# Create a merged DF, joining the metadata with the selected products with 5+ reviews in the a build set
md_merged = pd.merge(md[['asin','title','description']], df_mean_merged[['asin','overall']], on='asin')
print("\nmd_merged({}):\n".format(len(md_merged)), md_merged[:5])



# In[15]:


vectors_filepath = './data/vectors_each.{}.pytab'.format(build_set)
metadata_filepath = './data/metadata_each.{}.tsv'.format(build_set)
product_review_features_filepath = './data/product_features.{}.csv'.format(build_set)
product_attribute_features_filepath = './data/product_attribute_features.{}.csv'.format(build_set)


# In[16]:


# Preprocess the reviews by HTML unescaping them

#df_merged = df_merged.groupby(['asin','overall'], as_index=True)['reviewText'].apply(' '.join).apply(html.unescape).reset_index()
#print("cleaned df columns:", list(df))


# In[17]:


print("Cleaned DF", df_merged[:5])
print("...{} records".format(len(df_merged)))


# In[19]:


def remove_file(file_path):
    if os.path.isfile(file_path):
        print("Removing file '{}'...".format(file_path))
        os.remove(file_path)


# In[28]:


from string import punctuation

IGNORED_LEMMAS = ['-PRON-', 'PRON', 'i']
IGNORED_POS = ['PUNCT', 'DET']
MAX_FEATURES_PER_REVIEW = 10
MERGE_TOKENS = ["\'s", "\'t"]

debug = False

def get_word_scores(text_input, doc):
	if debug:
		print("Getting word scores for text:", text_input)
	tokenlist = [token for token in doc if token.norm_.lower() not in stopWords]
	wordlist = [token.norm_.lower() for token in tokenlist]
	wordfreq = []
	wordset = set(wordlist)
	for word in wordset:
	    wordfreq.append((word, np.log(wordlist.count(word)/len(wordlist))))
    
	s_word_freq = sorted(wordfreq, key=lambda pair : -pair[1])

	term_freq_dict = dict(zip(wordset,wordfreq))
	
	if debug:
		print("Sorted word frequency: ")
		for pair in s_word_freq:
			print(pair)

	word_scores = {}

	seen_tokens = set()

	for token in doc:
	    if token not in stopWords:
	        lower_norm = token.norm_.lower()
	        tf_pair = term_freq_dict.get(lower_norm)
	        if (tf_pair is not None) and (lower_norm not in seen_tokens):
	            tf = tf_pair[1]
	            word_scores[lower_norm]=tf-token.prob
	            seen_tokens.add(lower_norm)

	return word_scores


def merge_possible(prev_text, curr_text):
    """Determine whether tokens can be merged, if the current token is a punctuated suffix"""

    # First, check for a possessive "'s", after something other than punctuation
    if (curr_text == "\'s") and (prev_text[len(prev_text)-1] not in punctuation):
        return True
    # Next, check for a "'t" following an n
    elif (curr_text == "\'t") and (prev_text[len(prev_text)-1]=='n'):
        return True
    else:
        return False


def preprocess_doc(doc):
    """Preprocess spacy doc, searching for incorrectly split ['n','\'t'] and [?,'\'s'], merging and retokenizing doc"""
    merged = False
    

    prev_text = None
    token_ind = 0
    for token_ind in range(len(doc)):
        
        # The following is necessary because retokenization after merges shortens the list of tokens
        if token_ind == len(doc):
            break
        
        token = doc[token_ind]
        this_text = token.norm_.lower()
        if debug:
            print("text:{}, lem:{}, pos:{}, tag:{}, dep:{}, shp:{}, alph:{}, stop:{}, norm:{}".format(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, token.norm_))
        if (prev_text is not None) and (this_text in MERGE_TOKENS):
            if merge_possible(prev_text, this_text):
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(doc[(token_ind-1):(token_ind+1)], attrs={"NORM": prev_text+this_text})
                    merged = True
                    if debug:
                        print("Merged {} and {}".format(prev_text, this_text))
                    token_ind -= 1
                
                    
        prev_text = this_text
        
    return merged



def check_embedded_punctuation(input_str):
    """ Check for 2 sequential punctuation characters in a string """
    from string import punctuation
    
    punct_count = 0
    
    for char in input_str:
        if char in punctuation:
            punct_count += 1
            if punct_count == 2:
                return True
        else:
            punct_count = 0
            
    return False
        

def get_lemmatized_chunk(chunk):
    """ Filter a noun chunk to exclude IGNORED_LEMMAS and return the remaining text and a computed word vector. """
    processed_text = []
    vector = np.zeros(300)
    stop_count = 0
    non_stop_count = 0
    doc = chunk
    for token in doc:
        if (token.lemma_ not in IGNORED_LEMMAS) and (token.pos_ not in IGNORED_POS):            
            this_text = token.text.strip()
            
            # check if this token contains 2 or more sequential punctuation characters
            if check_embedded_punctuation(this_text):
                stop_count += 1
                continue
            else:  
                non_stop_count += 1
                vector = vector + token.vector

            #if this_text != token.text:
            #    processed_text.append(this_text)
            #else:
            if english_dict.check(token.norm_):
                processed_text.append(token.norm_.lower())   
    
    if non_stop_count > 3:
        return "", np.zeros(300)
    else:
        if (non_stop_count > 0) and (stop_count > 0):
            vector = np.divide(vector, non_stop_count)
    
    return " ".join(processed_text), vector


def get_vectors(text, nlp):
	""" <generator> Get embedding word vectors from a given text object. 
	Args
	----------
	text (string)            text to be parsed, tokenized, and vectorized
	nlp (spaCy pipeline)     pipeline to use for processing the input text
    
	Generates:
	----------
	processed text (string) 
	phrase vector (numpy.ndarray)
	"""          
	# first, strip out the stop words and lowercase the words
	text = ' '.join([word.lower() for word in text.split() if not word in stopWords])
    
	doc = nlp(text)

	# First, preprocess the text, retokenizing where spaCy has incorrectly split tokens
	preprocess_doc(doc)
	# Then, compute TF-IDF-style word scores for each word in the doc

	try:
		word_scores = get_word_scores(text, doc)
	except ValueError as ve:
		print("ERROR during word score computation for text:", text)
		raise ve
    

	#####
	# Next, iterate through the sentences and within those the noun chunks.
	# These noun chunks will be lemmatized and collected as potential features.
	#####
    
	collected_terms = []
	term_vector_map = {}
    
	for sent in doc.sents:
		for chunk in sent.noun_chunks:
		#yield chunk.text, chunk.vector
			lemmatized_text, vect = get_lemmatized_chunk(chunk)
			if len(lemmatized_text) >0:
				these_scores = []
				for word in lemmatized_text.split():
					this_score = word_scores.get(word)
					if this_score is not None:
						these_scores.append(this_score)
					#else:
					#	print("Token not found in word scores: This should never happen. Full text = '{}'. Lemmatized text = '{}', word = '{}'".format(text, lemmatized_text, word))
					#	print("Word scores:")
					#	for key, value in word_scores.items():
					#		print("    {}:    {}".format(key, value))
					#	raise ValueError("Token not found in word scores: This should never happen. Full text = '{}'. Lemmatized text = '{}', word = '{}'".format(text, lemmatized_text, word))

				if len(these_scores) == 0:				
					msg = "Processed text with no scores (probably all stop words). Full text = '{}', lemmatized text = '{}'".format(text, lemmatized_text)
					#print("Word scores:")
					#for key, value in word_scores.items():
				#		print("    {}:    {}".format(key, value))
					#raise ValueError(msg)
				else:
					collected_terms.append((lemmatized_text, np.mean(these_scores)))
					term_vector_map[lemmatized_text] = vect
                
	s_token_scores = sorted(collected_terms, key=lambda pair : -pair[1])
    
	for ranked_term in s_token_scores[:10]:
		term = ranked_term[0]
		yield term, term_vector_map[term]
        
def get_attribute_features(title, description, nlp):
    """ <generator> Get text features from a given product's title and description, in the same manner as for
    review text where features are the result of clustering
    
    Args
    ----------
    title (string)          text to be parsed, tokenized, and vectorized
    description (string)          text to be parsed, tokenized, and vectorized
    nlp (spaCy pipeline)    pipeline to use for processing the input text
    
    Generates:
    ----------
    processed text (string) 
    phrase vector (numpy.ndarray)
    """          
    
    if (title is not None) and (len(title)>0):
        for term, term_vector in get_vectors(title, nlp):
            yield term, term_vector
        
    if (description is not None) and (len(description)>0):
        for term, term_vector in get_vectors(description, nlp):
            yield term, term_vector
        


# In[33]:


test_collected_terms = []
test_term_vector_map = {}




doc = nlp(test_str)
for sent in doc.sents:
    for chunk in sent.noun_chunks:
        #yield chunk.text, chunk.vector
        lemmatized_text, vect = get_lemmatized_chunk(chunk)
        if len(lemmatized_text) >0:
            test_collected_terms.append(lemmatized_text)
            test_term_vector_map[lemmatized_text] = vect
            
print("test_collected_terms: ", test_collected_terms)

# In[30]:


def write_vectors(word_vects, out_m, product, rating, processed_text, concept_vec):
    """Write product, rating, phrase and sense vector to metadata and vectors files"""
    phrase = processed_text
    sense_vector = concept_vec
    print('\t'.join([product, str(rating),phrase]), file=out_m)
    word_vects.append([sense_vector])


# In[31]:


def write_product_attr_features(out_f, product, rating, processed_text, concept_vec):
    """Write product, mean rating, phrase and sense vector to product attribute features file"""
    phrase = processed_text
    sense_vector = concept_vec
    print(','.join([product, str(rating),phrase]), file=out_f)


# In[32]:


if non_clustered_product_features:
# Collect non-clustered features from each product's title and description

	remove_file(product_attribute_features_filepath)

	total_start = time.time()

	product_atts = md_merged

	print("There are {} total products with at least 5 reviews each".format(len(product_atts)))

	start_ind = 0
	iteration_size = 1000
	iter_limit = len(product_atts)
	features_count = 0

	print("\nCollecting word concept vectors for {} product titles & reviews...".format(iter_limit))
	display_local_time()

	with open(product_attribute_features_filepath, mode="a") as out_f:
    
	    for iteration in range(int(iter_limit/iteration_size)+1):

	        print("Starting iteration over products {}-{}...".format(start_ind + iteration*iteration_size,
	                                                                start_ind + (iteration+1)*iteration_size-1))
        
	        iter_start_time = time.time()
        
	        processed_records = iteration*iteration_size
        
	        this_iteration_size = min(iteration_size, len(product_atts)-processed_records)
        
	        for iter_ind in range(this_iteration_size):
    
	            product_ind = start_ind + iteration*iteration_size + iter_ind
        
	            product = product_atts['asin'].iloc[product_ind]
	            rating = product_atts['overall'].iloc[product_ind]
	            title = product_atts['title'].iloc[product_ind]
	            description = product_atts['description'].iloc[product_ind]
            
	            attr_text = ' '.join([str(title)+". ", str(description)])
        
	            if debug and (iteration == 0) and (iter_ind < 5):
	                print("product: {}, rating: {}, title: '{}', description: '{}'".format(product, rating, title, description))
    
	            #print(review)
	            for processed_text, concept_vec in get_vectors(attr_text, nlp):
            
	                # If there were no non-stop words in a given noun chunk, we will not add it to the vectors and metadata
	                if (len(processed_text)>0):

	                    write_product_attr_features(out_f, product, rating, processed_text, concept_vec)
	                    features_count += 1           

	        print("...completed processing {} products in {} seconds.".format(this_iteration_size, time.time()-iter_start_time))
    
	print("...processed {} products in {} seconds, producing {} total product_features.".format(iteration*iteration_size+this_iteration_size, time.time()-total_start, features_count))


# In[25]:


	product_attribute_features_filepath = "./data/product_attribute_features.0.csv"
	with open(product_attribute_features_filepath, mode="r") as in_f:
	    product_attr_features = pd.read_csv(in_f, names=['asin','overall','feature'])
	    print("File {} contains {} total product features.".format(product_attribute_features_filepath, 
	                                                               len(product_attr_features)))
	    print(product_attr_features[:20])
    


# In[25]:

if review_vectors:
# The following boolean controls whether the index list and the output np matrix are build by the vectorizing process.
# This can be used if appropriate for small datasets, but the repeated stacking of numpy arrays is expensive.
# Otherwise, these item are already written out to the files in metadata_filepath and vectors_filepath, using PyTables.
	build_vects = False

	remove_file(vectors_filepath)
	remove_file(metadata_filepath)

	# Create a sample vector, to determine the word vect dimension of a single entry
	sample_vect = [vec for vec in get_vectors("example", nlp)][0][1]
	vect_dim = sample_vect.shape
	print("Sample vect[{}]".format(vect_dim))
	index = []
	output = None
	vectors_count = 0

	total_start = time.time()

	#good_reviews = df[df['reviewerID'].isin(good_reviewers['reviewerID'])][df['asin'].isin(good_products['asin'])]
	#good_reviews = df[df['asin'].isin(good_products['asin']) & df['asin'].isin(products_in_buildset)]
	good_reviews = df_merged

	#print("There are {} total reviews for reviewers with at least 5 reviews each and products with at least 5 reviews each".format(len(good_reviews)))
	print("There are {} total reviews for products with at least 5 reviews each".format(len(good_reviews)))

	start_ind = 0
	iteration_size = 1000
	iter_limit = len(good_reviews)
	#iter_limit = 10000


	print("\nCollecting word concept vectors for {} product/rating/reviews...".format(iter_limit))
	display_local_time()

	with open_file(vectors_filepath, mode="w", title="Word Vectors") as out_v:
    
	    atom = tables.Float32Atom(vect_dim[0])
	    shape = (0,)
	    filters = tables.Filters(complevel=5, complib='zlib')
	    word_vect = out_v.create_earray(out_v.root, 'vector', atom, shape, filters=filters)

	with open(metadata_filepath, 'a') as out_m, open_file(vectors_filepath, mode="a", title="Word Vectors") as out_v:
    
	    print("Word Vectors: ", out_v)
    
	    for iteration in range(int(iter_limit/iteration_size)+1):

	        print("Starting iteration over reviews {}-{}...".format(start_ind + iteration*iteration_size,
	                                                                start_ind + (iteration+1)*iteration_size-1))
        
	        iter_start_time = time.time()
        
	        processed_records = iteration*iteration_size
        
	        this_iteration_size = min(iteration_size, len(good_reviews)-processed_records)
        
	        for iter_ind in range(this_iteration_size):
    
	            review_ind = start_ind + iteration*iteration_size + iter_ind
        
	            #reviewer = good_reviews['reviewerID'].iloc[review_ind]
	            product = good_reviews['asin'].iloc[review_ind]
	            rating = good_reviews['overall'].iloc[review_ind]
	            review = good_reviews['reviewText'].iloc[review_ind]
    
	            #print(review)
	            for processed_text, concept_vec in get_vectors(review, nlp):
            
	                # If there were no non-stop words in a given noun chunk, we will not add it to the vectors and metadata
	                if (len(processed_text)>0):

	                    write_vectors(out_v.root.vector, out_m, product, rating, processed_text, concept_vec)
	                    vectors_count += 1
                    
	                    # If this run is not just writing to disk, but should also build the vectors
	                    if build_vects:                    
	                        # Append data to a list and a numpy array
	                        index.append([product, rating, processed_text])
        
	                        if output is None:
	                            # Create an np.array with the first row as the retrieved word vector
	                            output = np.array([concept_vec])
	                        else:
	                            # Append the next vector to the end of the vectors array
	                            output = np.append(output, [concept_vec], axis=0)            

	        print("...completed processing {} reviews in {} seconds.".format(this_iteration_size, time.time()-iter_start_time))
    
	print("...processed {} reviews in {} seconds, producing {} word vectors.".format(iteration*iteration_size+this_iteration_size, time.time()-total_start, vectors_count))


# In[26]:

	phrases = []
	with open(metadata_filepath, 'r') as in_m:
	    for line in in_m:
	        phrase = line.split('\t')[2]
	        phrases.append(phrase.strip())
        
	phrase_count = Counter(phrases)

	print("Here are the top {} phrases and their counts...".format(MAX_FEATURES_PER_REVIEW))

	for counted_phrase in phrase_count.most_common(MAX_FEATURES_PER_REVIEW):
	        phrase = counted_phrase[0]
	        print(counted_phrase)


# In[ ]:


# Fit an HDBScan model using the sampled sense vectors
if clustering:
	HDBSCAN_METRIC = 'manhattan'

	with open_file(vectors_filepath, mode="r", title="Word Vectors") as word_vectors:

	    print("word_vectors: ", word_vectors)
    
	    start = time.time()
	    print("Creating word clusters from word vectors...")
	    display_local_time()
	    hdbscanner = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric=HDBSCAN_METRIC, gen_min_span_tree=True, core_dist_n_jobs=8, prediction_data=True)
	    hdbscanner.fit(word_vectors.root.vector)
	    print("...completed clustering in {} seconds.".format(time.time()-start))


# In[ ]:


	with open_file(vectors_filepath, mode="r", title="Word Vectors") as word_vectors:
	    vdim = len(word_vectors.root.vector)
	    print("word-vectors{}: ".format(vdim), word_vectors.root.vector)
	# Save the HDBScan model with a name indicating the number of word vectors clustered
	    with open('./data/hdbscanner.{}.pickle'.format(build_set), 'wb') as pickle_file:
	        pickle.dump(hdbscanner, pickle_file)


# In[ ]:


# Plot the condensed cluster tree

if plotting:
	start = time.time()
	print("Condensing the linkage tree and then plotting...")
	#hdbscanner.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
	hdbscanner.condensed_tree_.plot()
	hdbscanner.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
	print("...plotted condensed tree in {} seconds.".format(time.time()-start))
	tree = hdbscanner.condensed_tree_
	print("Found {} clusters".format(len(tree._select_clusters())))
	matplotlib.pyplot.show()


# In[ ]:


display_local_time()
print("Saving the CondensedTree to disk...")
start = time.time()
np.save("./data/condensedTree.{}.npy".format(build_set), hdbscanner.condensed_tree_)
print("...finished saving the CondensedTree in {} seconds.".format(time.time()-start))


# In[21]:


#print("Saving spacy language model to disk...")
#display_local_time()
#start = time.time()
#nlp.to_disk("./data/spacy.language.model.nlp")
#print("...finished saving spacy language model in {} seconds.".format(time.time()-start))


# In[24]:


def get_exemplars(cluster_id, condensed_tree):
    """ Collect and return the exemplar words for each cluster. """
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


# In[25]:


with open(metadata_filepath, 'r') as in_m, open("data/metadata_ouch.tsv", 'w') as out_m:
    item = 0
    for line in in_m:
        fields = line.split('\t')
        if len(fields) > 1:
            print("Fields: ", fields)
            phrase = fields[2]
            phrase = phrase[0:-1:2]
            print("Phrase: ", phrase.split())
            fields[:2]=phrase[:len(phrase)-1]
            out_line = '\t'.join(fields)
            print("Out line: ", out_line)
            item += 1
            if item == 10:
                break


# In[26]:
tree = hdbscanner.condensed_tree_

if labeling:
#print('Index, for reference:')
#for ind, entry in enumerate(index):
#    print("cluster: {}, ind: {}, entry: {}".format(hdbscanner.labels_[ind], ind, entry))

	start = time.time()
	print("Selecting clusters in tree...")
	clusters = tree._select_clusters()
	print("...finished selecting clusters in {} seconds.".format(time.time()-start))

	initial_cluster_count = len(clusters)
	print("Found {} clusters".format(initial_cluster_count))

	index = []
	all_points = []
	labels = []

# iterate through the input metadata once, to collect all words and the word labels for the sampled points
	with open(metadata_filepath, 'r') as in_m:
		mdim = None
		curr_line = 0
		for line in in_m:
			if mdim is None:
				mdim = line.count('\t')+1
				print('File {} contains index entries of of dimension {}'.format(metadata_filepath, vdim))
			if line.endswith('\n'):
				line = line[:-1]
			all_points.append(line.split('\t'))
			sample_this_row = True
			if sample_this_row:
				meta_line = line.split('\t')
				index.append(meta_line)
				if labels_words:
					labels.append(meta_line[2])
			curr_line += 1

	print("All Points({}, {}): {}".format(len(all_points), len(all_points[0]), all_points[:5]))

	# then, iterate through the input metadata again, to apply the cluster labels, if labels_words is False
	with open(metadata_filepath, 'r') as in_m:
		mdim = None
		curr_line = 0
		curr_sample = 0

		for line in in_m:
			if mdim is None:
				mdim = line.count('\t')+1
				print('File {} contains {} index entries.'.format(metadata_filepath, vdim))
			if line.endswith('\n'):
				line = line[:-1]
			sample_this_row = True
			if sample_this_row:
				index.append(line.split('\t'))
				if not labels_words:
					labels.append("-")
				curr_sample += 1
			curr_line += 1


# 

# In[27]:


	print(hdbscanner.exemplars_[:10])


# In[28]:


	print(nlp.vocab.vectors.most_similar(hdbscanner.exemplars_[1]))


# In[29]:


	selected_clusters = []
	cluster_map = {}
	cluster_exemplar_map = {}

	for i, c in enumerate(clusters):
	    c_exemplars = get_exemplars(c, tree)

	    point_label = None
	    cluster_exemplars = set()
	    for ind, ex_ind in enumerate(c_exemplars):
	        #print("Exemplar -- {} : {}".format(index[ex_ind][0], index[ex_ind][2]))
	        
	        candidate_exemplar = index[ex_ind][2]
        
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
    
	    # Look for clusters where the members outnumber the exemplars by 2 times
	    if ((exemplars_len>0) and (len(members)>(2.0*exemplars_len))):
    
	        example_cluster_exemplars = "|".join(cluster_exemplars)
	        example_cluster_members = "|".join(members)
        
	        selected_clusters.append((example_cluster_exemplars, example_cluster_members))
	        # with the index of the cluster (treated as label by hdbscanner) as a key, store the index into selected_clusters
	        cluster_map[str(i)] = len(selected_clusters)-1
	        cluster_exemplar_map[str(i)] = cluster_exemplars

	selected_cluster_count = len(selected_clusters)
	if (selected_cluster_count>0):
	    with open("./data/clusters.{}.txt".format(build_set), "w") as cluster_report:
	        print("\nFound {} clusters ({}% of initially collected):".
	          format(len(selected_clusters), 100.0*float(selected_cluster_count)/float(initial_cluster_count)), file=cluster_report)
	        for example in selected_clusters:
	            print("\nExemplars: {}".format(example[0]), file=cluster_report)
	            print("Members: {}".format(example[1]), file=cluster_report)

	with open(path_for_tf_metadata+'/metadata.{}.tsv'.format(build_set), 'w') as out_tf_meta:
	    for label in labels:
	        out_tf_meta.write(str(label)+'\n')
                                                                    
	noise_count = sum([1 for label in hdbscanner.labels_ if label == -1])
	print("\nThere were {} words that were considered noise.".format(noise_count))

	np.save('./data/selected_clusters.{}.npy'.format(build_set), selected_clusters)


# In[30]:


	print(hdbscanner.labels_[:10])


# In[31]:


def get_scored_exemplars(phrase, cluster_id):
    """ TODO -- Given a phase and a cluster_id (the label from hdbscanner for the phrase) return a set of scored exemplars.
    
    Note that this function doesn't yet return the results it should. The cell below does produce the right results though, and should be used instead.
    """
    scored_exemplars = {}
    exemplars = cluster_exemplar_map.get(str(cluster_id))
    if exemplars is None:
        return None
        exemplars = get_exemplars(cluster_id, hdbscanner.condensed_tree_)
        cluster_exemplars[str(cluster_id)]=exemplars
        
    phrase_doc = nlp(phrase)
    for exemplar in exemplars:
        exemp_doc = nlp(exemplar)
        scored_exemplars[exemplar]=phrase_doc.similarity(exemp_doc)
        
    return scored_exemplars

clustered_labels = [labels for labels, _ in cluster_exemplar_map.items()]

for ind, label in enumerate(hdbscanner.labels_):
    product = index[ind][0]
    phrase = index[ind][2]
    cluster_id = label
    
    if (cluster_id >= 0) and (cluster_id in clustered_labels):
        
        print("Product: {}, phrase: '{}', scored_exemplars: {}".format(product, phrase, 
                                                                       get_scored_exemplars(phrase, cluster_id,)))


# In[32]:


print("selected_clusters:", selected_clusters[:5])
print("cluster_map:", [item for item in cluster_map.items()][:5])

def get_score(exem_tuple):
    return -exem_tuple[1]

if text_features:
	with open(product_review_features_filepath, 'w') as prod_features_file:
    
	    # create an empty dict in which to hold phrases we've already seen associated with a product
	    visited_product_phrases = {}
    
	    for ind, cluster_ind in enumerate(hdbscanner.labels_):
	        # A non-negative hdbscanner label for a point indicates assignment to a cluster
	        if cluster_ind >= 0:
	            cluster_detail_ind = cluster_map.get(str(cluster_ind))
	            if cluster_detail_ind is None:
	                continue
	            else:
	                pass
	                #print("Found detail for cluster {} : {}".format(cluster_ind, cluster_detail_ind))
	            cluster_detail = selected_clusters[cluster_detail_ind]
	            product = index[ind][0]
	            rating = index[ind][1]
	            phrase = index[ind][2]
            
	            # see if we've already seen this phrase in this product, if so skip it
	            already_visited_list = visited_product_phrases.get(product)
	            if already_visited_list is None:
	                visited_product_phrases[product]=[phrase]
	            else:
	                if phrase in already_visited_list:
	                    continue
	                else:
	                    already_visited_list.append(phrase)
	                    visited_product_phrases[product] = already_visited_list
                    
	            phrase_doc = nlp(phrase)

	            exemplars = cluster_detail[0]
	            scored_exemplars = []
	            for exemp in exemplars.split("|"):
                

                
	                exemp_doc = nlp(exemp)
	                ex_similarity = phrase_doc.similarity(exemp_doc)
	                scored_exemplars.append((exemp, ex_similarity))
	            scored_exemplars = sorted(scored_exemplars, key=get_score)
        
            #print("product:{}, rating:{}, phrase:'{}', cluster:{}, exemplars:{}".format(product, rating, phrase, cluster_ind, scored_exemplars))
	            print("{}, {}, {}, '{}'".format(product, rating, cluster_ind, scored_exemplars[0][0], scored_exemplars), file=prod_features_file)


# In[62]:


# This ends up being a ***TON*** of labels, with the full run


#for ind, label in enumerate(hdbscanner.labels_[:10]):
#    if label >= 0:
#        print("ind: {}, product: {}, label: {}".format(ind, index[ind][0], label))
            
        


# In[19]:


# Prepare for a tensorboard visualization

if tensor_vis:
	import tensorflow as tf

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


# In[20]:


# Generate the tensorboard embedding visualization
# To view it, run command "tensorboard --port=6006 --logdir=./logdir" on your computer and then 
# open http://localhost:6006 in a browser.

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


# In[21]:


def generate_sample_vectors(text, nlp):
    """ Preprocess test text content, for predicting labels."""
    
    sample_index = []
    sample_vectors = None
    
    for concept_vec in get_vectors(text, nlp):
            
        # Append data to a list and a numpy array
        sample_index.append([product, rating, concept_vec[0]])
        
        if sample_vectors is None:
            # Create an np.array with the first row as the retrieved word vector
            sample_vectors = np.array([concept_vec[1]])
        else:
            # Append the next vector to the end of the vectors array
            sample_vectors = np.append(sample_vectors, np.array([concept_vec[1]]), axis=0)            
    
    return sample_index, sample_vectors


# In[24]:



def get_sample_labels(cluster_ind):
    sample_labels = set()
    for word_ind in get_exemplars(clusters[cluster_ind], tree):
        exemplar_label = html.unescape(index[word_ind][2])
        sample_labels.add(exemplar_label)
        
    return sample_labels


# In[25]:


sample_text = "I would buy this again. It was a very good deal and good for children 2-4. Usually, this is good fun and good for the whole family, brothers, sisters and friends. Another with a magnetic personality was buying sweets for his sweet. Something should be done to handle all shapes and sizes and enable then to be put together."

sample_index, sample_vectors = generate_sample_vectors(sample_text, nlp)

test_labels, strengths = hdbscan.approximate_predict(hdbscanner, sample_vectors)
print("test_labels: ",test_labels)

for ind, word_index in enumerate(sample_index):
    if test_labels[ind] >0:
        print("Phrase '{}' is predicted to have labels {}.".format(word_index[2], get_sample_labels(test_labels[ind])))


# In[ ]:




