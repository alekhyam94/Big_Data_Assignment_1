import pickle
import numpy as np
import torch
import time
import torchvision
import matplotlib
import matplotlib.pyplot as plt

def load_cifar_data(data_files):
    data = []
    labels = []
    for file in data_files:
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            if len(data) == 0:
                data = data_dict[str.encode('data')]
                labels = data_dict[str.encode('labels')]
            else:
                data = np.vstack((data, data_dict[str.encode('data')]))
                labels.extend(data_dict[str.encode('labels')])
    return data, labels

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res
    
def get_classwise_indices(labels):
    label_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_indices.keys():
            label_indices[label] = [idx]
        else:
            label_indices[label].append(idx)
    return label_indices
            
def get_data_from_indices(data, indices_dict, count_per_class, image_shape):
    generated_data = []
    generated_labels = []
    for key, val in indices_dict.items():
        if count_per_class:
            for i in range(count_per_class):
                generated_data.append(np.reshape(data[val[i]], image_shape))
                generated_labels.append(key)
        else:
            for i in val:
                generated_data.append(np.reshape(data[i], image_shape))
                generated_labels.append(key)
    return np.asarray(generated_data), np.reshape(np.asarray(generated_labels, dtype=np.int32), (-1,1))

def create_data_loader(data_x, data_y, batch_size, shuffle):
    tensor_x = torch.stack([torch.Tensor(i) for i in data_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in data_y])

    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create datset
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle) # create dataloader
    return dataloader
    
def train_model(model, train_data_loader, test_data_loader, num_epochs=5, learning_rate=0.001, save_epochs=None, model_name="cnn"):
    num_epochs = num_epochs
    learning_rate = learning_rate

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_step = len(train_data_loader)
    train_times = []
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_data_loader):
            # Forward pass
            outputs = model(images)
            target = torch.max(labels.long(), 1)[0]
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        end_time = time.time()
        if save_epochs and epoch + 1 in save_epochs:
            torch.save(model, "../data/models/" + model_name + "_" + str(epoch+1))
        train_times.append(end_time - start_time)
        train_losses.append(loss.item())       
        print("Calculating train accuracy...")
        train_accuracies.append(get_accuracies(train_data_loader, model)[0])
        print("Calculating test accuracy...")
        test_accuracies.append(get_accuracies(test_data_loader, model)[0])
    print("Average training time per epoch:", np.mean(train_times))
    print("Total training time for all epochs:", np.sum(train_times))
    return train_accuracies, test_accuracies, train_losses

def get_accuracies(data_loader, model):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            labels = torch.max(labels.long(), 1)[0]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    end_time = time.time()
    time_taken = end_time - start_time
    print('Accuracy of the model: {} %'.format(accuracy))
    return accuracy, time_taken
    
def get_model_size(model, model_name):
    model = pickle.dumps(net)
    byte_size = sys.getsizeof(model)
    print('Size of ' + model_name + ' model: ', byte_size/1000000)
    
def imshow(img, label_names, file_name="../data/sample_images"):
    npimg = img.numpy()
    npimg = npimg.astype(np.uint8)
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.clf()
    im = plt.imshow(npimg)
    ylim = im.get_extent()[2]
    plt.yticks(np.arange(0, ylim + 1, ylim/len(label_names)), label_names)
    plt.savefig(file_name)
    plt.show()
    
def show_classwise_images(data, labels, label_names, k):
    image_dict = {}
    for idx, l in enumerate(labels):
        label = l[0]
        if label in image_dict.keys() and len(image_dict[label]) < k:
            image_dict[label].append(data[idx])
        elif label not in image_dict.keys():
            image_dict[label] = [data[idx]]
    
    images_to_show = []
    labels_to_show = []
    for label, image in image_dict.items():
        labels_to_show.append(label_names[label])
        for i in image:
            images_to_show.append(i)
    
    images_tensor = torch.stack([torch.Tensor(i) for i in images_to_show])
        
    imshow(torchvision.utils.make_grid(images_tensor, nrow=k), labels_to_show)
    
def outlier_analysis(model, outliers_tensor, outlier_label_names, cifar10_label_names):
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        start_time = time.time()
        outputs = model(outliers_tensor)
        end_time = time.time()
        print("Time taken for prediction:", str(end_time - start_time))
        _, predicted = torch.max(outputs.data, 1)
        for idx, label in enumerate(predicted):
            print("Original:", outlier_label_names[idx], "Predicted:", cifar10_label_names[label])
            predicted_labels.append(cifar10_label_names[label])
    imshow(torchvision.utils.make_grid(outliers_tensor, nrow=1), predicted_labels)
    
def plot_values(x, y, xlabel, ylabel, title, legend, fig_name):
    plt.clf()
    for y_i in y:
        plt.plot(x, y_i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig("../data/plots/" + fig_name)
    plt.show()