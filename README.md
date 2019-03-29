# Eliciting New Users' Interests in Wikipedia

This repository includes the code to perform the following:

1. Perform pre-processing on the user-document and document-term data.
2. Generate user-document and document-term matrices.
3. Perform SVD.
4. Perform joint topic extraction from user and document data.
5. Generate questions.
6. Perform offline evaluation.
7. Prepare the data needed for online deployment.

## What do I do to run the pipeline?

Here, the whole pipeline of the project from the initial files to the final questions and online deployment material is listed. The formats of the data that are assumed to be provided are given in the next section.

Each of the scripts listed below has a usage message that will be output if you run them with no arguments.

1. create_doc_edit_histograms.py : Creates histograms of document edit count, user edit count, and document length, while filtering out:
   - Documents whose names fall within "list/list-like" criteria (defined within the code).
   - Documents of yearly events (e.g. 2005 Tour de France, etc.)
   - Bots
   - Administrative pages
   A list of bots and a list of non-administrative pages are provided to the script. The name criteria are defined as regular expressions in split_and_filter.py, in the function discard_docs_by_names. *Requires Apache Spark*.
2. create_keep_lists.py : Creates "keep lists" of users and docs. These are actually not lists of users and docs that we will certainly keep, but anything not in them we will certainly discard. Here, documents with their lengths below a certain threshold (defined in constants.py) are discarded. The script will also show you the average number of edits for each user and document after having discarded the stubs.
3. create_user_doc_matrix.py : Creates a user-doc matrix using the revision history data and the keep lists you've generated in the previous step. Also uses thresholds from constants.py to filter out documents and users with too few edits *iteratively*. It uses the function iterative_keeplist_filtering in split_and_filter.py. You can choose to do a train/test split here, or provide 0 as the respective argument - which results in all data becoming training data. *Requires Apache Spark*.
4. pairs_tf_idf.py : Creates a set of (doc_id, word_id, TFIDF_score) tuples using document content data (word counts), while only keeping the documents provided in a dictionary. The dictionary must be the column dictionary (col_dict.json) produced by step #3. *Requires Apache Spark*. This step and step #6 can be bypassed if you simply have a TF-IDF document-term matrix from somewhere. Just remember to filter out rows that are not keys of the column dictionary.
5. matrix_rdd_to_csr.py : Converts an rdd matrix in the format (ID, SparseVector), like the one(s) produced by step #3, into a scipy CSR matrix. If you have done a train/test split, then you will have to run this once for each of the matrices. *Requires Apache Spark*.
6. pairs_to_csr.py : Converts the TF-IDF tuples to a CSR matrix. *Requires Apache Spark*.
7. create_holdout_set.py : Only if you have done a train/test split, run this on the test set to generate a holdout set for offline evaluation. *Requires Apache Spark*.
8. extract_singular_vecs.py : Performs an SVD on the matrix provided as input. Use this with the document-term matrix to get a doc latent matrix (which we will call the original doc latent matrix). If you train a word2vec or sent2vec model on the documents that we're keeping in the end, you can skip #4, #6 and this step and use the resulting word2vec/sent2vec model as the original doc latent matrix.
9. recommender_minibatchgd.py : The joint topic extraction step. Here, you will provide:
    - The user-document matrix
    - The original document latent matrix (produced by #10)
    - The user keep list by entropy (if applicable)
    - The hyperparameters
    What you get is a doc_latent.pickle matrix, which is the resulting document latent matrix of our full method. Other settings of the gradient descent are in minibatch_settings.json.
10. create_questions.py : Using a doc latent matrix (doesn't matter which one), produces the questions that are the output of our method. This script creates a json dictionary and an avoid_list pickle file, which will be used for the online system.
11. If you're doing offline evaluation (having performed a train/test split and holdout set creation before), you should run offline_nonadaptive_test.py here. 

Congratulations, you've made it to the end! At step #10, a text file is also output, and you can see the questions in that file.

### Before you start
Have a look at the file utils/constants.py; this file contains many of the parameters that you need to set in order to run the pipeline. Some of these are constants that need to _remain_ constant (e.g. USER_ID, PAGE_ID which are column indices for an input file). Some, such as USER_INACTIVITY_THRESH, DOC_INACTIVITY_THRESH, and DOC_STUB_THRESH, can be set using feedback from the pipeline (e.g. the histograms from step #1 above). Others are just design choices, such as DISCARD_YEAR_LISTS. The defaults are the parameters that have been used to generate the results in the paper.

### Notes

* Originally, our textual data came from a pre-processing pipeline that would create parallel corpora of Wikipedia articles in multiple languages, but the data we have made available to you is only for the English Wikipedia, and therefore, any "concept-doc file" you see in the arguments for the scripts is redundant and you should use -none as the input for them.
* For the hyperparameters of the minibatch gradient descent, consult the paper.

## What format should the data be in?

You need the following files:
* A TSV (tab-separated values) file with the schema (article_id, word_id, count). This is for the textual content of Wikipedia.
* A TSV file with the schema (word_id, 0, word). This is also for the textual content.
* A TSV file for the revision history of Wikipedia which has the user id, user name, and article id. The default column indices for these three (in the data provided to you) are in utils/constants.py .
* A TSV file with the schema (article_id, article_namespace, article_name). This file is to filter out articles outside namespace 0 (actual articles) and also to map ids to names.
* A text file containing Wikipedia bot names (one bot name per file), to filter them out.

## Where is this data?

The data is located at TODO.

## What do I need for the online system?

* The doc latent matrix produced by the minibatch gradient descent script.
* An article id to article name map (JSON file).
* An article id to article index map (JSON file). The indices are for the rows of the doc latent matrix.
* The questions.json file created by create_questions.py.
* The id avoid list created by create_questions.py (to avoid recommending articles that appeared in the questionnaire itself).
