import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image as im
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import copy


# SET SEED
seedNum = 3
torch.manual_seed(seedNum)
np.random.seed(seedNum)

# DATA PREPARATION
def get_mnist_data():
    data_train = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True)
    data_test = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True)
    
    x_train, y_train = data_train.train_data.numpy().reshape(-1, 1, 28, 28),  np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1, 1, 28, 28), np.array(data_test.test_labels)
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def get_cifar10_data():
    data_train = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
    
    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)),  np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def get_fashionMnist_data():
    data_train = torchvision.datasets.FashionMNIST(root='./data/fashionMNIST', train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root='./data/fashionMNIST', train=False, download=True)
    
    x_train, y_train = data_train.train_data.numpy().reshape(-1, 1, 28, 28),  np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1, 1, 28, 28), np.array(data_test.test_labels)
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes,x_train, y_train,x_test, y_test)

def get_transform_function(name_dataset):
    transforms_func = {
            'mnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.06078,), (0.1957,))
            ]),
            'fashionmnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'cifar10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        }
    return transforms_func[name_dataset]

def transform_image_array(images,labels,name_dataset):
    transform = get_transform_function(name_dataset)
    
    normalized_images = torch.zeros(images.shape)
    for i,image in enumerate(images):
        image = image.astype(float)
        normalized_images[i] = transform(np.transpose(image, (1, 2, 0)))
    
    return normalized_images,torch.from_numpy(labels)

def image_reshape(image_array,dim):
    new_dim = [image_array.shape[1]]+dim
    new_image_array = np.zeros([len(image_array)]+new_dim)
    for index,image in enumerate(image_array):
        image_resized = np.transpose(image,(2,1,0))
        if new_dim[0] == 1:
            image_resized = im.fromarray(np.reshape(image_resized,image_resized.shape[0:2]))
            image_resized = ImageOps.mirror(image_resized.rotate(270))
            image_resized = image_resized.resize(new_dim[1:])
            image_resized = np.reshape(np.array(image_resized),new_dim)
        else:
            image_resized = im.fromarray(image_resized)
            image_resized = image_resized.resize(new_dim[1:])
            image_resized = np.transpose(np.array(image_resized),(2,1,0))
        new_image_array[index] = image_resized
    return new_image_array.astype(int) if new_dim[0] == 3 else new_image_array

def shuffle_image_array(image,label):
    shuffle_index = np.linspace(0,len(image)-1,len(image)).astype(int)
    np.random.shuffle(shuffle_index)
    
    return image[shuffle_index], label[shuffle_index]

def split_data(images,labels,num_clients,shuffle,name_dataset,dim):
    extra = False
    images = image_reshape(images,dim)
    image_shape = images.shape
    
    if num_clients>len(images):
        print("Impossible Split!!")
        exit()
        
    if shuffle:
        images, labels = shuffle_image_array(images, labels)
    
    client_list = []
    for i in range(num_clients):
        client_list.append("client_"+str(i+1))
    
    images_normalized,labels_tensor = transform_image_array(images,labels,name_dataset)
    
    if(len(images)%num_clients != 0):
        extra_images = len(images)%num_clients
        extra = True
    
    len_data_per_clients = len(images)//num_clients
    Data_Split_Dict = {} #Client_name: (image,label)
    it = 0
    for index,name in enumerate(client_list):
        if index%2 == 1:
            array_split = images_normalized[it:it+(len_data_per_clients//2)]
            label_split = labels_tensor[it:it+len_data_per_clients//2]
            Data_Split_Dict[name] = [array_split,label_split]
            it = it + len_data_per_clients//2
        if index%2 == 0:
            array_split = images_normalized[it:it+(len_data_per_clients+len_data_per_clients//2)]
            label_split = labels_tensor[it:it+(len_data_per_clients+len_data_per_clients//2)]
            Data_Split_Dict[name] = [array_split,label_split]
            it = it + len_data_per_clients+len_data_per_clients//2            
    
    client_names = [k for k,v in Data_Split_Dict.items()]
    if extra:
        for i, (image,label) in enumerate(zip(images_normalized[-1*extra_images:],labels_tensor[-1*extra_images:])):
            new_data = torch.reshape(image,(-1,image.size()[0],image.size()[1],image.size()[2]))
            Data_Split_Dict[client_names[i%num_clients]][0] = torch.cat((Data_Split_Dict[client_names[i%num_clients]][0],new_data),dim=0)
            
            label_list = torch.reshape(Data_Split_Dict[client_names[i%num_clients]][1],(-1,1))
            new_label = torch.reshape(label,(-1,1))
            Data_Split_Dict[client_names[i%num_clients]][1] = torch.flatten(torch.cat((label_list,new_label),dim=0))
    
    return client_names,Data_Split_Dict

def collect_batch(data,label,batch_num,batch_size):
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    
    if batch_num == batch_count and extra != 0:
        batch = (data[batch_num*batch_size:],label[batch_num*batch_size:])
    else:
        batch = (data[batch_num*batch_size:batch_num*batch_size+batch_size],label[batch_num*batch_size:batch_num*batch_size+batch_size])
    if batch_num >= batch_count:
        batch = (-1,-1)
        
    return batch

#MODIFICATION HERE############################################################################################################################
#MODIFICATION HERE############################################################################################################################
# MODEL CREATION
class Net(nn.Module):
    def __init__(self, num_class, dim):
        super(Net, self).__init__()
        self.num_class = num_class
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Dimension after convolutional layers
        conv_dim = dim[0] // 4 * dim[1] // 4 * 64
        
        # Dense layers
        self.fc1 = nn.Linear(conv_dim, 256)
        self.fc2 = nn.Linear(256, num_class)
  
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# TRAINING
def train_local(model,data,label,client_name,epoch,batch_size):
    optimizer = Adam(model.parameters(), lr=0.000075)
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        print("Cuda Activated")
    
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    print("{} {} training starts!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    
    for e in range(epoch):
        print("*",end=" ")
        training_losses = []
        for b in range(batch_count):
            batch_data,batch_label = collect_batch(data,label,b,batch_size)
            
            batch_label = batch_label.type(torch.LongTensor)
            if torch.cuda.is_available():
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs,batch_label)
            
            training_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            accuracy = torch.sum(torch.max(outputs,dim=1)[1]==batch_label).item() / len(batch_label)

    print("")
    training_loss = np.mean(training_losses)    
    #print("\nLast Epoch!!! \t training loss: {} \t accuracy:{}".format(training_loss,accuracy))
    #print("{} {} training ends!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    
    return model #training_loss

def validation(model,test_data,test_label,dim,name_dataset):
    val_x,val_y = copy.deepcopy(test_data), copy.deepcopy(test_label)
    val_x = image_reshape(val_x,dim)
    val_x, val_y = transform_image_array(val_x,val_y,name_dataset)
    
    if torch.cuda.is_available():
        val_x = val_x.cuda()
        model = model.cuda()

    with torch.no_grad():
        outputs = model(val_x)
        
    overall_accuracy = torch.sum(torch.max(outputs.cpu(),dim=1)[1]==val_y).item() / len(val_y)
    
    return overall_accuracy,torch.max(outputs.cpu(),dim=1),val_y


# SYNCRONIZATION & AGGREGATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def syncronize_with_server(server, client):
    target = {name:value.to(device) for name,value in client.named_parameters()}
    source = {name:value.to(device) for name,value in server.named_parameters()}
    
    for name in target:
        target[name].data = source[name].data.clone()

def federated_averaging(server, clients):
    target = {name:value.to(device) for name,value in server.named_parameters()}
    sources = []
    for client_name,client_model in clients.items():
        source = {name:value.to(device) for name,value in client_model.named_parameters()}
        sources.append(source)
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


#MODIFICATION HERE############################################################################################################################
#MODIFICATION HERE############################################################################################################################
def client_data_sizes(Data_Split_Dict):
    clients_name_to_data_sizes = {client:len(data[1]) for client, data in Data_Split_Dict.items()}
    return clients_name_to_data_sizes


#MODIFICATION HERE############################################################################################################################
#MODIFICATION HERE############################################################################################################################
def client_participation_rates(clients_name_to_data_sizes):
    participation_rates = torch.tensor([data_size for _, data_size in clients_name_to_data_sizes.items()])
    participation_rates = torch.div(participation_rates,torch.sum(participation_rates))
    clients_participation_rates = {client_name:round(participation_rates[i].clone().item(), 6) for i, client_name in enumerate(clients_name_to_data_sizes.keys())}
    return clients_participation_rates, participation_rates


#MODIFICATION HERE############################################################################################################################
#MODIFICATION HERE############################################################################################################################
def weighted_average(server, clients, clients_name_to_data_sizes):
    target = {name:value.to(device) for name,value in server.named_parameters()}
    sources = []
    for client_name,client_model in clients.items():
        source = {name:value.to(device) for name,value in client_model.named_parameters()}
        sources.append(source)
    _, participation_rates = client_participation_rates(clients_name_to_data_sizes)

    for name in target:
        each_client_model_weight = torch.stack([source[name].data for source in sources])
        
        result = torch.zeros_like(each_client_model_weight[0])  # Create a tensor to store the sum of multiplied matrices
        for i in range(each_client_model_weight.shape[0]):
            result += participation_rates[i] * each_client_model_weight[i]
        target[name].data = result.clone()


   