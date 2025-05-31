# Code accompanying the paper 'Hierarchical recurrent temporal prediction as a model of the mammalian dorsal visual pathway'

## Directory structure

  1. Root
     * `submit_training_job.py` - code to submit a 'job' to train a model via the command line
     * `train.py` - code to train the model
     * `utils.py` - utility functions
  2. `analysis` - Jupyter notebooks to analyse the trained models
  3. `data` - contains the `dataset.py` data loader, and a place to store training datasets
  4. `losses` - contains files defining the different loss functions
  5. `model_checkpoints` - place to store model checkpoints during training
  6. `models` - contains files defining the different models
  7. `neural_fitting` - contains response similarity analysis code including code to download datasets from the Allen Brain Institute
  8. `virtual_physiology` - code to run 'virtual physiology' on the trained models

## Getting started

  1. Set up
     * Add the path and name to your training dataset at `train.py:22`
         * A small example dataset (`example_dataset.npy`) is provided in the data folder
         * Datasets should be stored as a numpy array of dimensions (number of clips, number of frames, flattened number of pixels)
     * Set the training dataset name in `submit_training_job.py:45-47`, loss name (e.g., loss_hierarchical_temporal_prediction) and model name (e.g., network_hierarchical_recurrent)
  2. Run `submit_training_job.py --name [name] --L1 [L1 strength]`, where:
      * `--name` is the name to save the checkpoint as
      * `--L1` is the log L1 regularization strength applied during training (e.g., -6)
  4. When the model has finished training (this may take some time), run the virtual physiology
     * For each folder in `virtual_physiology`, give the location of the model checkpoint and the location to save the processed virtual physiology 'artefact' (you can give the same virtual physiology path across these files)
     * To run all the virtual physiology procedures, you can use `run.bash`
 5. Analyze the trained models using the `analysis` folder. This should be as easy as providing the relevant model, virtual physiology and dataset paths where indicated in the notebooks
