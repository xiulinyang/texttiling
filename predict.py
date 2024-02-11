import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
import json
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
import json
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from nltk.metrics import windowdiff
import re
import torch.nn.functional as F
from transformers import BertModel
from sklearn.metrics import classification_report
from transformers import DebertaV2Model, AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model =DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")



INPUT=''
MAX_LEN=400 #maximum length of the input
loss_fn = nn.BCEWithLogitsLoss()  # don't change
EPOCH = 2
HIDDEN_SIZE=200
LEARNING_RATE = 5e-6
BS = 4 #batch size
DROPOUT =0.05
EPSILON = 1e-8
WINDOW_SIZE=2
NUM_WARMUP = 1000
TRAINING_SAMPLE=1000000

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

MODEL_SAVE_PATH = "roberta_new2.pth"

def preprocess_data(text_list):
    processed_text =''
    for text in text_list:
        if text.split() and text.split()[0].isupper():
            continue
        pattern = r'[\u0000-\u001f\u007f-\u009f]'
        # Replace the matched characters with an empty string
        cleaned_text = re.sub(pattern, '', text)
        processed_text+=cleaned_text
    return processed_text

def preprocess(text):
    return text


def data_preprocess(data, context_window):
    inputs = []
    labels = []
    with open(data) as data_file:
        data_f = json.load(data_file)
        print(len(data_f))
        for i, f in enumerate(data_f):
            sentence = f['sentence']
            if i + context_window < len(data_f):
                sentencea = data_f[i+1]['sentence']
                sentenceaa= data_f[i+2]['sentence']
            else:
                sentencea = ' <eod>'
                sentenceaa = ' <eod>'
            if i<context_window:
                input = '<sod> '* (context_window-i) + '<sos>'+sentence + '<eos>' + sentencea + sentenceaa

            else:
                sentencep = data_f[i-1]['sentence']
                sentencepp = data_f[i-2]['sentence']
                input = sentencepp+' '+sentencep+ '<sos>'+sentence + '<eos>' + sentencea + sentenceaa
            if i>TRAINING_SAMPLE:
                break
            label = label_num_pair[int(f['boundary-coarse'])]
            inputs.append(input)
            labels.append(label)
    return inputs, labels


def preprocessing_for_bert(input):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(input):
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

        encoded_sent = tokenizer.encode_plus(
            text=preprocess(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

class BertClassifier(nn.Module):
    def __init__(self, model):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out =768, HIDDEN_SIZE, 1

        # Instantiate BERT model
        self.model = model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            #nn.ReLU(),
            #nn.Dropout(DROPOUT),
            #nn.Linear(H, H),
            #nn.ReLU(),
            #nn.Dropout(DROPOUT),
            #nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        first_token_tensor = last_hidden_state[:, 0]
        logits = self.classifier(first_token_tensor)
        #pooled_output = outputs.pooler_output
        #logits = self.classifier(pooled_output)
        return logits


def initialize_model(model, epochs=EPOCH):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(model)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=LEARNING_RATE,    # Default learning rate
                      eps=EPSILON    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(dev_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=NUM_WARMUP,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.float().unsqueeze(1))
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                #if evaluation == True:
                    # After the completion of each training epoch, measure the model's performance
                    # on our validation set.
                    #val_loss, val_accuracy = evaluate(model, val_dataloader)
                    #_, train_accuracy = evaluate(model, train_dataloader)
                    # Print performance over the entire training data
                    #time_elapsed = time.time() - t0_epoch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^7}  | {'-':^7} | {'-':^7} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            _, train_accuracy = evaluate(model, train_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels.float().unsqueeze(1))
        val_loss.append(loss.item())

        # Get the predictions
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        # Convert probabilities to binary predictions (0 or 1)
        accuracy = (preds == b_labels.cpu().numpy()).mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy




def bert_predict(model, test_dataloader):
    gold_labels =[]
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    all_logits = []
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        gold_labels.extend(b_labels.cpu().numpy())
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
            # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # Apply softmax to calculate probabilities
    probs = torch.sigmoid(all_logits).cpu().numpy()
    # Convert probabilities to binary predictions (0 or 1)
    predictions = (probs > 0.5).astype(int).flatten()  # Threshold is 0.5
    final_report = classification_report(gold_labels, predictions, target_names=['no', 'yes'])
    g = [str(x) for x in gold_labels]
    p = [str(x) for x in predictions]
    print('=============Windowdiff metric======================')
    print(windowdiff(g, p, 8))
    print('=============Classification report===================')
    print(final_report)
    with open('report_bert.txt', 'a') as report:
        report.write(final_report)
    return predictions

if __name__=='__main__':
    set_seed(42)
    # prepare data
    label_num_pair = {0:0, 1:1}
    dev_input, dev_labels = data_preprocess('New-York-City.json',WINDOW_SIZE)

    dev_input_ids, dev_attention_masks = preprocessing_for_bert(dev_input)

    # Convert other data types to torch.Tensor
    dev_labels_ID = torch.tensor(dev_labels)


    # Create the DataLoader for our training set

    # Create the DataLoader for our validation set
    dev_data = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels_ID)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BS)




    bert_classifier, optimizer, scheduler = initialize_model(model,epochs=EPOCH)

    model_load_path = MODEL_SAVE_PATH
    checkpoint = torch.load(model_load_path, map_location=device)
    bert_classifier.load_state_dict(checkpoint['model_state_dict'])
    bert_classifier.to(device)  # Move model to the right device
    bert_classifier.eval()  # Set the model to evaluation mode for predictionsptimizer.load_state_dict(checkpoint['optimizer_state_dict']t
    pred = bert_predict(bert_classifier, dev_dataloader)
    with open('model_preds.txt', 'w') as wf:
        wf.write(str(pred))
