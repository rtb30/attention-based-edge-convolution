import os
import os.path as osp
from os import makedirs
import importlib
import torch
from pantomime_dataset import PantomimeDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse
import numpy as np
import sys
from utils.augmentation_transformer import AugmentationTransformer
import torch_geometric.transforms as Transformers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

torch.manual_seed(3407)
# neural network, particularly those using cuDNN (NVIDIA’s deep neural network library), behave deterministically
#torch.backends.cudnn.deterministic = True

BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))

########################### MODEL VARIABLES ###########################
nearest_neighbors   = 20          # 24 is the limit
conv_layers         = 2
max_epoch           = 1000
gpu                 = 0
batch_size          = 30
num_gestures        = 21
early_stop_flag     = 'True'
early_stop_count    = 75
learning_rate       = 0.001       # moderate rates are 0.001 or 0.0001, schedulers help exponentially decay learning rates

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='edgecnn',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: modified_edgecnn]')
parser.add_argument('--log_dir', default='stgcnn', help='Log dir [default: stgcnn]')
parser.add_argument('--k', default=nearest_neighbors, type=int, help='Number of nearest points [default: 4]')
parser.add_argument('--t', default=1000, type=int, help='Number of future frames to look at [default: 1]')
parser.add_argument('--spatio_temporal_factor', default=0.01, type=float, help='Spatio-temporal factor [default: 0.01]')
parser.add_argument('--graph_convolution_layers', default=conv_layers , type=int, help='Number of graph convolution layers [default: 21]')
parser.add_argument('--max_epoch', type=int, default=max_epoch, help='Epoch to run [default: 251]')
parser.add_argument('--normalize_data', default=False, help='Normalize the point cloud [default: False]')
parser.add_argument('--gpu_id', default=gpu, help='GPU ID [default: 0]')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size [default: 32]')
parser.add_argument('--dataset', default='data/primary_32_f_32_p_without_outlier_removal', help='Dataset path. [default: data/primary_32_f_32_p_without_outlier_removal]')
parser.add_argument('--num_class', type=int, default=num_gestures, help='Number of classes. [default: 21]')
parser.add_argument('--early_stopping', default=early_stop_flag, help='Whether to use early stopping [default: True]')
parser.add_argument('--early_stopping_patience', type=int, default=early_stop_count,
                    help='Stop the training if there is no improvements after this ' +
                         'number of consequent epochs [default: 100]')
 
FLAGS = parser.parse_args()
DATASET = FLAGS.dataset
LOG_DIR = FLAGS.log_dir
K = FLAGS.k
T = FLAGS.t
SPATIO_TEMPORAL_FACTOR = FLAGS.spatio_temporal_factor
GRAPH_CONVOLUTION_LAYERS = FLAGS.graph_convolution_layers
GPU_ID = FLAGS.gpu_id
MAX_EPOCH = FLAGS.max_epoch
NORMALIZE_DATA = FLAGS.normalize_data
MODEL = importlib.import_module(FLAGS.model)
BATCH_SIZE = FLAGS.batch_size
NUM_CLASSES = FLAGS.num_class
EARLY_STOPPING = FLAGS.early_stopping
EARLY_STOPPING_PATIENCE = FLAGS.early_stopping_patience

if not osp.exists(LOG_DIR):
    print('Creating the model checkpoint directory at {}'.format(LOG_DIR))
    makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

model_path = osp.join(LOG_DIR, 'model.pth')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()

device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), DATASET)
augmentation_transformer = AugmentationTransformer(False, BATCH_SIZE)

if NORMALIZE_DATA:
    pre_transform = Transformers.NormalizeScale()
    train_dataset = PantomimeDataset(path, True, pre_transform=pre_transform)
    test_dataset = PantomimeDataset(path, False, pre_transform=pre_transform)
else:
    train_dataset = PantomimeDataset(path, True)
    test_dataset = PantomimeDataset(path, False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# set the model
model = MODEL.Net(NUM_CLASSES, 
                  graph_convolution_layers = GRAPH_CONVOLUTION_LAYERS, 
                  k = K, 
                  T = T, 
                  spatio_temporal_factor = SPATIO_TEMPORAL_FACTOR).to(device)

#model = MODEL.Net(NUM_CLASSES, 
#                  graph_convolution_layers = GRAPH_CONVOLUTION_LAYERS, 
#                  k = K).to(device)

# optimize and schedule learning rate for model progression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

########################## PLOT CONFUSION MATRIX #########################
# Assuming 'conf_matrix' is the confusion matrix returned by your 'test' function
def plot_confusion_matrix(conf_matrix, class_labels, 
                          output_path="attention-based-edge-convolution/ConfusionMatrix_Pics/confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    # Save the plot as a PNG file
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

########################### LOAD TRAINING DATA ###########################
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        #print(f'Training batch {i}...')
        #data = augmentation_transformer(data)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.squeeze().to(torch.long))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    scheduler.step()
    return total_loss / len(train_dataset)

########################### LOAD TESTING DATA ############################
def test(loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y.squeeze()).sum().item()

        all_preds.append(pred.cpu().numpy())
        all_labels.append(data.y.cpu().numpy().squeeze())

    accuracy = correct / len(loader.dataset)
    
    # Convert list of arrays to single arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Print the unique labels to see the order
    unique_labels = np.unique(all_labels)
    #print("Unique labels in the test set:", unique_labels)
    
    # Calculate additional metrics
    # set zero_division = 0 to surpress warning when class size is small due to random batch feeds with smaller batch sizes
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, conf_matrix, unique_labels

############################ MAIN SCRIPT: ACCURACY LOGGING #############################
log_string('\n--------------------- SELECTED MODEL ---------------------\n{}'.format(MODEL))
log_string('\n**********************************************************')
log_string('Batch Size    : {}'.format(batch_size))        
log_string('Conv Layers   : {}'.format(conv_layers))
log_string('Learning Rate : {}'.format(learning_rate))
log_string('**********************************************************\n')

best_acc = -1
best_acc_epoch = -1
best_train_loss = 0
current_acc = -1
last_improvement = 0
for epoch in range(1, MAX_EPOCH):
    loss = train()                                                      # augmentation happens here
    current_acc, precision, recall, f1, conf_matrix, class_labels = test(test_loader) # find testing data accuracy

    ######################### PRINT BEST ACCURACY #########################
    if current_acc > best_acc:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}'.format(epoch, loss, precision, recall, f1))

        log_string('********************** Test Accuracy: {:.4f} **********************\n'.format(current_acc))

        plot_confusion_matrix(conf_matrix, class_labels)

        #if(current_acc > 0.55):
        #    log_string('Confusion Matrix:\n{}'.format(conf_matrix))

        torch.save(model.cpu().state_dict(), model_path)  # saving model
        #model.cuda() # moves model to GPU
        #log_string('The model saved in {}'.format(model_path))

        best_acc = current_acc
        best_acc_epoch = epoch
        best_train_loss = loss
        last_improvement = 0

    ################ PRINT ACCURACY IF BEST ACCURACY EXISTS ################
    elif best_acc > 0:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {:.4f}, Best Test Accuracy: {:.4f}, Best Epoch: {}'.format
                   (epoch, loss, current_acc, best_acc, best_acc_epoch))

        last_improvement += 1

    ####################### FIRST PRINTED ACCURACY ########################
    else:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, loss, current_acc))

        last_improvement += 1

    ########################### EARLY STOPPING ############################
    if EARLY_STOPPING == 'True' and last_improvement > EARLY_STOPPING_PATIENCE:
        log_string('\n*************************** MODEL EARLY STOP INITIATED ***************************')
        log_string('- No improvement was observed after {} epochs.'.format(last_improvement))
        log_string('The best model with {:.1f} percent accuracy, {:.2f} loss, was saved at epoch {}.'.format(
            (best_acc * 100), loss, best_acc_epoch))
        break

if EARLY_STOPPING != 'True':
    log_string('The best model with {:.1f} percent accuracy, {:.2f} loss, was saved at epoch {}.'.format(
        (best_acc * 100), loss, best_acc_epoch))