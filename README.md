# Chatbot

## Setup

From the project directory, run

        pip3 install -r requirements
        
## Running the server

Run

        cd Flaskapp
        python3 __init__.py

Afterwards, navigate to localhost:5001 in your browser.

## Interacting with a basic chatbot

If no network has been trained, the bot is configured
to answer by repeating your question. This should work
out of the box.

## Training the recurrent sequence-to-sequence model

To train the recurrent network, download the dataset from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
and move its contents into the project directory at

        ./training_data/cornell movie-dialogs corpus

Also, you should create the folder

        ./saved_models/
        
(do not commit the training data nor the saved 
network states to the repo)
        
Now, run the network training script via

        python3 ./train.py



