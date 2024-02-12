# -----------------------------------------------------------------------------
# This script file provides fundamental computational functinality for performing
# sentiment classificaion on the Musk Tweets Hourly dataset. The sentiment 
# classifier to be built will be utilizing the BERT model which will be used for
# tokenization and word-embedding purposes. The final model will be trained on 
# the go emotions database.
# -----------------------------------------------------------------------------

# Import required Python libraries.
import openpyxl
import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel

# -----------------------------------------------------------------------------
#                          FUNCTION DEFINITIONS:
# -----------------------------------------------------------------------------
# This method provides fundamental text cleaning functionality. 
def clean_text(text):
    # Remove emoticons
    emoticon_pattern = re.compile(u'('
        u'\ud83d[\ude00-\ude4f]|'  # emoticons
        u'\ud83c[\udf00-\uffff]|'  # symbols & pictographs (part 1)
        u'\ud83d[\u0000-\uddff]|'  # symbols & pictographs (part 2)
        u'\ud83d[\ude80-\udeff]|'  # transport & map symbols
        u'\ud83c[\udde0-\uddff]'   # flags (iOS)
        u')+', flags=re.UNICODE)
    text = emoticon_pattern.sub(r'', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Lowercase the words
    text = text.lower()
    return text
# -----------------------------------------------------------------------------
# This method provides fundamental parsing functionality for a given excel 
# file. The output returned is a list of strings where each string contains 
# a comma separated character array with the cell contents for each row of the
# original excel file.
def parse_excel(file_path):
    # Initialize the list of strings to be returned by this function.
    lines = []
    # Load the workbook and select the active sheet.
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    # Iterate over the various rows of the file and read each cell as text.
    for row in sheet.iter_rows():
        # Initialize the current line container to the empty string.
        line_text = ""
        # Iterate over the contents of the current row.
        for cell in row:
            # Convert the content of each cell to string so as to be appended
            # to the current row.
            cell_value = str(cell.value) if cell.value is not None else ""
            line_text += cell_value + ","
        # Append current line string to the lines list.
        lines.append(line_text)
    return lines
# -----------------------------------------------------------------------------
#                          CLASS DEFINITIONS:
# -----------------------------------------------------------------------------
# Define the Custom Dataset Class.
# -----------------------------------------------------------------------------
class SentimentDataset(Dataset):
    
    # Class constructor.
    def __init__(self,texts,labels,tokenizer,max_length):
        self.texts = [clean_text(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    # Implement the custom len() behaviour for the objects of this class.
    def __len__(self):
        return len(self.texts)
    
    # Implement the class method that allows slicing within the contents of the 
    # custom dataset class.
    def __getitem__(self,idx):
        text = self.texts[idx]
        if len(self.labels)>0:
            label = self.labels[idx]
        # ---------------------------------------------------------------------
        # Encoding text technical details:
        # ---------------------------------------------------------------------
        # This is a method provided by the BERT tokenizer loaded from the 
        # Transformers library. It takes a text string and converts it into 
        # input IDs and attention masks necessary for BERT.
        # ---------------------------------------------------------------------
        # Input Parameters:
        # ---------------------------------------------------------------------
        # text: The text to be encoded. This is the individual text string 
        #       from the dataset that needs sentiment analysis.
        # add_special_tokens = True: This tells the tokenizer to add special 
        #                            tokens like [CLS] at the beginning and 
        #                            [SEP] at the end of each text string. 
        #                            These tokens are important for BERT as 
        #                            [CLS] is often used for classification 
        #                            tasks, and [SEP] is a separator token, 
        #                            useful especially when dealing with two 
        #                            text strings (like in question-answering 
        #                            models.
        # ---------------------------------------------------------------------
        # max_length: This sets the maximum length of the tokenized input 
        #             sequence. If the text is longer than this, it will be 
        #             truncated to max_length. This value is typically set to 
        #             the maximum length the model can accept (e.g., 512 tokens 
        #             for BERT).
        # ---------------------------------------------------------------------
        # return_token_type_ids = False: This is specific to certain BERT tasks
        #                                that requires differentiating between
        #                                multiple input sequences like questing
        #                                answering tasks. For simple classification
        #                                tasks this value should be set to False.
        # ---------------------------------------------------------------------
        # padding = 'max_length': This ensures that all encoded sequences are
        #                         padded to the same length (max_length). In 
        #                         case a sequence is shorter than max_length,
        #                         it will be padded with zeros.
        # ---------------------------------------------------------------------
        # return_attention_mask = True: This directive instructs the tokenizer
        #                               to generate and return attention masks,
        #                               which tell the model which tokens should
        #                               be attended to and which should not
        #                               (e.g. padding tokens)
        # ---------------------------------------------------------------------
        # return_tensors = 'pt': This specifies that the returned tensors should
        #                        PyTorch tensors.
        # ---------------------------------------------------------------------
        # trancation = True: This argument ensures that if a text string is 
        #                    longer than the max_length, it will be truncated
        #                    to fit.
        # ---------------------------------------------------------------------
        # Output Parameters:
        # ---------------------------------------------------------------------
        # input_ids: These are the token ids for each token in the text.They
        #            constitute the input for the BERT model.
        # ---------------------------------------------------------------------
        # attention_mask: This is a mask of 1s and 0s indicating which tokens 
        #                 are actual words and which are padding.The BERT 
        #                 model utilizes this information to know which parts 
        #                 of the input it should pay attention to and which 
        #                 parts should be ignored.
        # ---------------------------------------------------------------------
        
        # Encoding text.
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_length,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
            truncation = True
            )
        
        # Create the dictionary object which will be returned as the item for 
        # the requested position.
        # Check whether text sentiment labels are not provided:
        if len(self.labels)==0:
            item = {
                'text' : text,
                'input_ids' : encoding['input_ids'].flatten(),
                'attention_mask' : encoding['attention_mask'].flatten()
                }
        else:
            item = {
                'text' : text,
                'input_ids' : encoding['input_ids'].flatten(),
                'attention_mask' : encoding['attention_mask'].flatten(),
                'label' : torch.tensor(label,dtype=torch.long)
                }
        return item
# -----------------------------------------------------------------------------

# Define an additional function that will preserve a given percentage of data
# instances per class in order to reduce the overall size of the dataset in 
# order to reduce the computation time required to process it.This function 
# will return the data indices to be retained keeping an equal amount of 
# patterns from each distinct class.
def balanced_dataset_reduction(dataset,percentage):
    # Initialize a dictionary object that will be used for grouping the data
    # points indices per class.
    indices_by_class = defaultdict(list)
    
    # defaultdict: is a class provided by the collections module in Python. 
    # It works like a regular dictionary but with one key difference: it 
    # automatically creates new items for keys that have not been encountered 
    # yet. When you try to access a key that doesn't exist, instead of raising a
    # a KeyError, defaultdict creates the key and initializes it to a default 
    # value that you specify.
    
    # list: in this case is the default value type specified for the 
    # defaultdict. It means that when a new key is accessed in indices_by_class
    # that doesn't exist yet, a new empty list will be created as its value.
    
    # So, indices_by_class = defaultdict(list) creates a defaultdict where the 
    # default value type for new keys is an empty list. This is useful for 
    # collecting indices corresponding to different class labels in a dataset, 
    # as you can simply append indices to the list associated with each class 
    # label without having to check if the key exists first.
    
    # Loop through the various datapoints stored in the dataset object by 
    # keeping both the actual data and the corresponding indices.
    for idx,data in enumerate(dataset):
        # Acquire the class label for the currently processed data instance.
        label = data["label"].item()
        indices_by_class[label].append(idx)
        
    # Shuffle and select indices from each class.
    # Initialize the list container that will store the data indices that will
    # be retained.
    filtered_indices = []
    # Loop through the value entries for each key in the dictionary.
    for indices in indices_by_class.values():
        np.random.shuffle(indices)
        # Compute the number of indices to be kept.
        retain_number = int(len(indices)*percentage)
        filtered_indices.extend(indices[:retain_number])
    return filtered_indices
# -----------------------------------------------------------------------------
# Define the Sentiment Classifier Class. 
# -----------------------------------------------------------------------------
class SentimentClassifier(nn.Module):
    
    # Class constructor.
    def __init__(self,bert_model,n_classes,dropout_percent=0.3):
        # Call the super class constructor.
        super(SentimentClassifier,self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=dropout_percent)
        self.out = nn.Linear(self.bert.config.hidden_size,n_classes)
        
    # Define the function that describes the forward pass of information within
    # the network. Mind that the network module is being fed with the input ids
    # and the attention mask provided by the BERT tokenizer for each text.
    def forward(self,input_ids,attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
            )
        # When return_dict=False, the BERT model returns a tuple of two elements:
        # [1]: The hidden state of all layers of the neural model.
        # [2]: The pooled output, which is the representation of the input 
        #      sequence after being processed by the model.
        # By setting return_dict=False, we are explicitly instructing the BERT 
        # model to return a tuple of outputs instead of a dictionary. This is 
        # often done for compatibility with older versions of the Hugging Face 
        # Transformers library or for custom model implementations that expect 
        # tuples as outputs. If return_dict=True, the BERT model would return a 
        # dictionary containing various outputs, such as the hidden states, 
        # pooled output, and other intermediate outputs. 
        output = self.drop(pooled_output)
        return self.out(output)
    
# -----------------------------------------------------------------------------
#                         INITIALIZATION TASKS:
# -----------------------------------------------------------------------------

# Read the individual data files into dataframe objects.
df1 = pd.read_csv("data/goemotions_1.csv")
df2 = pd.read_csv("data/goemotions_2.csv")
df3 = pd.read_csv("data/goemotions_3.csv")

# Concatenate the individual dataframes into a single dataframe object.
df = pd.concat([df1,df2,df3],ignore_index=True)

# Delete the constituent dataframes.
del df1,df2,df3

# Filter all examples that have been marked as very unclear.
unclear = df["example_very_unclear"]==True
df = df[~unclear]

# Drop columns that are not required for the sentiment analysis task.
dropped_columns = ["id","author","subreddit","link_id","parent_id",
                   "created_utc","rater_id","example_very_unclear"]
df.drop(dropped_columns,axis="columns",inplace=True)

# Isolate the list of string texts.
texts = list(df["text"])

# Isolate the dataframe of labels and covert it to a numpy array.
labels = df.iloc[:,1:].to_numpy()
# Convert the one-hot vector encoded labels into a list of integer ids 
# indicating the corresponding sentiment category.
labels = list(np.argmax(labels,axis=1))
# Get the number of classes existing within the ground truth dataset.
classes_num = len(set(labels))

# Set the location of the dataset that will be exclusively used for testing
# purposes and for which no ground truth labels are available.
excel_file_path = 'data/Musk_HOURLY.xlsx'
# Read the excel lines.
excel_lines = parse_excel(excel_file_path)
# Read the text of each tweet which is stored in the excel file.
# Assuming that the textual content of each line is a comma separated string,
# we need to keep the first substring. 
excel_texts = [line.split(",")[0] for line in excel_lines]
# Drop the first entry since it corresponds to the content of the header line.
excel_texts = excel_texts[1:]
# Additionally, we need to define a lambda function which returns the last
# non-empty string in a list of strings.
last_non_empty = lambda lst: next((s for s in reversed(lst) if s),None)
# Thus, we can acquire the date string for each tweet knowing that it is the
# last piece of information which is stored in each excel line.
excel_dates = [last_non_empty(line.split(",")) for line in excel_lines]
# Drop the first entry since it corresponds to the content of the header line.
excel_dates = excel_dates[1:]

# -----------------------------------------------------------------------------
#                    LOAD BERT TOKENIZER AND BERT MODEL:
# -----------------------------------------------------------------------------

# Set the name of the BERT model to be utilized.
model_name = 'bert-base-uncased'
# Instantiate the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained(model_name)
# Instantiate the BERT model.
bert_model = BertModel.from_pretrained(model_name)

# -----------------------------------------------------------------------------
#                           DATA PREPARATION:
# -----------------------------------------------------------------------------

# Set the maximum length for the input sequences.
max_length = 512

# Set the batch size to be used during the training process.
batch_size = 2

# Create the ground truth dataset instance of the SentimentDataset class.
ground_truth_dataset = SentimentDataset(texts,labels,tokenizer,max_length)

# Set the percentage of data to be kept from each distinct class.
retain_percent = 0.50

# Filter the ground truth dataset per class according to the prespecified 
# percentage value per class.
filtered_indices = balanced_dataset_reduction(ground_truth_dataset,retain_percent)

# Generate the reduced version of the complete dataset.
reduced_ground_truth_dataset = torch.utils.data.Subset(ground_truth_dataset,filtered_indices)

# Create the dataset instance for the unseen set of tweets.
unseen_dataset = SentimentDataset(excel_texts,[],tokenizer,max_length)

# Set the percentage of the available data that will be used for training.
train_percent = 0.8

# Set the train and the test sizes.
train_size = int(train_percent * len(reduced_ground_truth_dataset))
test_size = len(reduced_ground_truth_dataset) - train_size

# Split the ground truth dataset into training and testing subsets.
train_dataset, test_dataset = random_split(
    reduced_ground_truth_dataset, [train_size,test_size])

# Instantiate the training and testing data loaders.
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size)


# -----------------------------------------------------------------------------
#                          MODEL EVALUATION PROCESS:
# -----------------------------------------------------------------------------

def evaluate_model(model,data_loader,device,return_probabilities=False):
    # Set the model evaluation environment.
    model.eval()
    # Initialize variables counting the total and correctly classified text
    # instances.
    total,correct = 0,0
    # Initialize the list containing the output probabilities for each text
    # instance.
    all_probabilities = []
    
    # Indicate that no gradient-based updating of the model weight-vector will
    # be performed during this process.
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Acquire the information that the bert tokenizer associated with
            # each textual input.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Acquire the network output for each instance in the batch.
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            # Convert the network primitive output to probabilities.
            probabilities = torch.nn.functional.softmax(outputs,dim=1)
            # Get the predicted class index by determining which component of
            # the probability vector contains the maximum value.
            _,predicted = torch.max(probabilities,1)
            # If the corresponding input argument indicates that probability
            # vectors should also be returned, then perform the necessary 
            # computation.
            if return_probabilities:
                all_probabilities.extend(probabilities.cpu().numpy())
            # Accumulate the total number of text instances that have been 
            # classified so far.
            total += labels.size(0)
            # Accumulate the total number of text instances that have been 
            # correctly classified.
            correct += (predicted==labels).sum().item()
    
    # Compute the total accuracy of the model on the subset of text instances
    # stored in the data loader.
    accuracy = correct / total
    
    return (accuracy,np.array(all_probabilities)) if return_probabilities else accuracy 

# -----------------------------------------------------------------------------
#                          MODEL TRAINING PROCESS:
# -----------------------------------------------------------------------------

def train_model(model,train_loader,test_loader,optimizer,loss_fn,epochs,
                checkpoint_path=None,save_every_batches=None):
    
    # Load the state variables from the last training session.
    checkpoint = torch.load(checkpoint_path)
    # Load the last state of the neural model.
    model.load_state_dict(checkpoint["model_state_dict"])
    # Load the last state of the optimizer.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Set the starting epoch of the training process to be the last 
    # training epoch of the previous session.
    start_epoch = checkpoint["epoch"]
    # Load the last batch processed during the previous training session.
    batch_count = checkpoint["batch_count"]
    # Load the best accuracy achieved by the model so far.
    best_accuracy = checkpoint["best_accuracy"]
    
    # Actual Model Training Process
    
    # Loop through the remaining training epochs.
    for epoch in range(start_epoch,epochs):
        # Set the model training environment.
        model.train()
        # Loop through the various batches.
        for batch_idx,batch in enumerate(tqdm(train_loader,desc=f"Epoch: {epoch+1} / {epochs}")):
            # Skip batches until the last processed batch is reached.
            if batch_idx < batch_count:
                continue
            # Load the necessary information from the current batch.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Compute the output of the model and the associated loss by 
            # performing the forward pass of information.
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            loss = loss_fn(outputs,labels)
            # Optimize model parameters by performing the backward pass of 
            # information.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Increase the number of batches processed so far.
            batch_count += 1
            # Save checkpoint if specified and if the currently processed batch
            # is an integer multiple of the save_every_batches input arguments.
            # Thus, a training checkpoint will be saved after every given amount
            # of batches has been processed.
            if save_every_batches is not None and batch_count % save_every_batches==0:
                torch.save({
                    'epoch':epoch,
                    'batch_count':batch_count,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'best_accuracy':best_accuracy
                    },checkpoint_path)
    
    # Evaluate the model on the test set after each epoch.
    accuracy = evaluate_model(model,test_loader,device)
    
    # Save final checkpoint after each training epoch has been completed.
    torch.save({
        'epoch':epoch,
        'batch_count':batch_count,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'best_accuracy':accuracy
        },checkpoint_path)
    
    # Report the termination of the training process.
    print("Training process completed")

# -----------------------------------------------------------------------------
#    CLASSIFIER, LOSS FUNCTION, OPTIMIZER AND CHECK POINT INITIALIZATION:
# -----------------------------------------------------------------------------
model = SentimentClassifier(bert_model,classes_num)
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
# Set the number of training epochs.
epochs = 20
# Set the device on which training will be performed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Method .to(device) is a PyTorch method used to move tensors or models to a 
# specified device. The device argument specifies where the tensors or model 
# should be moved. It could be either a CPU or a GPU. In the context of deep 
# learning, moving models to GPUs can significantly speed up computation due to 
# their parallel processing capabilities.

# Define the name of the checkpoint directory
checkpoint_directory = "checkpoints"
# Define the name of the file that will actually store the checkpoint dictionary.
checkpoint_filename = "model_checkpoint.pth"
# Create the full path for the checkpoint directory within the current directory
checkpoint_path = os.path.join(checkpoint_directory,checkpoint_filename)
# Set the number of batches within each training epoch after which a checkpoint
# entry will be saved.
batches_save_period = 5
# Create the checkpoint directory and the corresponding file in case it does 
# not exist. In that case, create the file and save the initial state of the
# training process.
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
    f = open(checkpoint_path,"w")
    f.close()
    # Set the initial state variables of the training process.
    epoch = 0
    batch_count=0
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    best_accuracy = 0
    torch.save({
        'epoch':epoch,
        'batch_count':batch_count,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'best_accuracy':best_accuracy
        },checkpoint_path)

# Call the model training method.
train_model(model,train_loader,test_loader,optimizer,loss_fn,epochs,
            checkpoint_path=checkpoint_path,save_every_batches=batches_save_period)