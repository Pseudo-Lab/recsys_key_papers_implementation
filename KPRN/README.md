# knowledge-graph-recommender
Comps replication repository on knowledge graphs for recommendation. We implemented the KPRN model described in https://arxiv.org/abs/1811.04540 on subnetworks of the KKBox song dataset, compared it to a Matrix Factorization baseline, and added extensions to the paper's model. Our results are described in our paper: kprn_replication.pdf.

Python version is 3.6.3, and packages can be installed via `./install_requirements`

## KPRN General Usage Information
To train and evaluate the KPRN model, you can either use our random sample rs subnetwork, whose kg files are already in the github repository (skip to recommender.py command line arguments), or create a subnetwork yourself. 

To construct the kg yourself first download the `songs.csv` and `train.csv` from https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data. Then create a folder called `song_dataset` in `knowledge-graph-recommender/data` and place `songs.csv` and `train.csv` in `song_dataset`. These files are larger than the github limit.

Then construct the knowledge graph with data-preparation.py, and path-find, train, and evaluate using recommender.py.

### Knowledge Graph Construction
Run data-preparation.py to create relation dictionaries from the KKBox dataset

Command line arguments:

`--songs_file` to specify path to CSV containing song information (default is songs.csv)

`--interactions_file` to specify path to CSV containing user-song interactions (default is train.csv)

`--subnetwork` to specify data to create knowledge graph from. Options are dense, rs, sparse, and full.
In our project we used the dense and rs versions, where dense contains the top 10% entities with highest degree, and rs contains a random 10% sample of entities.


### recommender.py command line arguments

`--subnetwork` to specify subnetwork training and evaluating on.

`--train` to train model, `--eval` to evaluate

`--find_paths` if you want to find paths before training or evaluating

`--kg_path_file` designates the file to save/load train/test paths from

`--user_limit` designates the max number of train/test users to find paths for (larger limit will improve results)

`--model` designates the model to train or evaluate from

`--load_checkpoint` if you want to load a model checkpoint (weights and parameters) before training

`--not_in_memory` if training on entire dense subnetwork, whose paths cannot fit in memory all at once

`--lr`, `--l2_reg`, `--gamma` specify model hyperparameters (learning rate, l2 regularization, weighted pooling gamma)

`-b` designates model batch size and `-e` number of epochs to train model for

Note: For evaluating, subnetwork and weighted pooling gamma must be the same as they were set to for training.

### Training syntax
Command line syntax to find train paths and train model on 100 users on rs subnetwork:

Note: each positive interaction is paired with 4 negative ones.

`python3 recommender.py --train --find_paths --user_limit 100 --kg_path_file train_inters_rs_100.txt --model rs_100.pt -e 10 --subnetwork rs`

### Evaluating syntax
Command line syntax to find test paths and evaluate trained model(the 100 user saved model) on 100 users:

Note: Each interaction group is 1 positive interaction paired with 100 negative interactions.

`python3 recommender.py --eval --find_paths --user_limit 100 --kg_path_file test_inters_100.txt --model rs_100.pt --subnetwork rs`
