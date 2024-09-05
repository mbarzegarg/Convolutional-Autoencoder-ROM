"""
@author: Sajad Salavati
"""
#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import glob
import matplotlib.tri as tri
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import imageio
from typing import Tuple, Union
from torchsummary import summary
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
import copy
from torchvision import transforms
import torchvision
import torch.nn.init as init
from scipy.linalg import norm

#%% Run Info
#Exercise 1
T  = 1
dt = 0.0005
dt_in_T = int(T / dt)
n_snapshots = dt_in_T
#%% Interpolating BC
state_BC = 'unnecessary' # save |unnecessary

if state_BC == 'save':
    
    df = pd.read_csv('BC.csv')
    
    #Velocity Function Interpolation
    tv_values = df['t1'].tolist()
    v_values = df['v'].tolist()
    #Create interpolation function
    interp_func = interp1d(tv_values, v_values)
    # Generate 2000 uniform values
    uniform_tv_values = np.linspace(min(tv_values), max(tv_values), n_snapshots)
    interpolated_v_values = interp_func(uniform_tv_values)
    
    #Pressure Function Interpolation
    tp_values = df['t2'].tolist()
    tp_values = [x for x in tp_values if not math.isnan(x)]
    P_values = df['P'].tolist()
    P_values = [x for x in P_values if not math.isnan(x)]
    #Create interpolation function
    interp_func = interp1d(tp_values, P_values)
    #Generate 2000 uniform values
    uniform_tp_values = np.linspace(min(tp_values), max(tp_values), n_snapshots)
    interpolated_P_values = interp_func(uniform_tp_values)

elif state_BC == 'unnecessary':
    print('transformed BC will be read next')
    
#%% Importing data
state_file_read = 'load' # save | load

if state_file_read == 'save':

    u = {}
    
    # Define the path to your CSV files
    csv_path = 'Table_Fluid_table_*.csv'

    # Get a list of all the CSV files matching the pattern
    csv_files = sorted(glob.glob(csv_path))

    # Extract the table numbers from the file names
    table_numbers = [int(file.split('_')[-1].split('.')[0]) for file in csv_files]

    csv_files_sorted = [file for _, file in sorted(zip(table_numbers, csv_files))]

    # Read X, Y, and Z columns from the first CSV file
    df_first = pd.read_csv(csv_files[0])
    X = df_first['X (m)'].values
    Y = df_first['Y (m)'].values
    
    # Iterate over each CSV file
    for file in csv_files_sorted:
        # Extract the table number from the file name
        table_number = int(file.split('_')[-1].split('.')[0])
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Store each column in the corresponding matrix
        u[table_number] = df['Velocity[j] (m/s)'].values
        print(table_number)

    u = np.vstack(list(u.values()))

    
    row_avg = np.mean(u, axis=1)
    fig, ax1 = plt.subplots()
    ax1.plot(row_avg, color='red', label='A')
    ax1.plot(interpolated_v_values, color='blue', label='B')
    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(interpolated_P_values, color='green', label='C')
    plt.show()

    np.save("S1_variable_and_BC.npy", (u, X, Y, interpolated_v_values, interpolated_P_values))
        
    
elif state_file_read == 'load':
    u, X, Y, interpolated_v_values, interpolated_P_values = np.load("S1_variable_and_BC.npy", allow_pickle=True)
    print('u and transformed BC are loaded')
a
#%% Save Pictures
state_load_image = 'unnecessary' # save | unnecessary
if state_load_image == 'save':
    max_radius = 0.005
    ii = 1
    for i in range (0, len(u)):
        #####################################Creating Grid######################################
        # Create the Triangulation; no triangles so Delaunay triangulation created.
        triang = tri.Triangulation(X, Y)
        levels = np.arange(0., 1., 0.025)
        triangles = triang.triangles
        
        # Mask off unwanted triangles.
        xtri = X[triangles] - np.roll(X[triangles], 1, axis=1)
        ytri = Y[triangles] - np.roll(Y[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
        triang.set_mask(maxi > max_radius)
        
        u0 = u[i,:].T / 0.1
        levels = np.linspace(0, 1, 100)
        
        
        # Close any existing figures
        plt.close('all')
        # Turn off interactive mode
        plt.ioff()
        # Create a figure and set the size explicitly
        fig1, ax1 = plt.subplots(figsize=(8, 6))  # Adjust the size as desired
        
        # Plot the tricontourf
        tcf = ax1.tricontourf(triang, u0, cmap="gray", levels=levels, vmin=0, extend='both')
        
        # Hide the axes
        ax1.set_axis_off()
        
        # Remove the extra white space around the plot
        ax1.margins(0)
        
        # Save the figure without excess margins
        image = 'u' + str(i) + '.png'
        plt.savefig(image, dpi=200, bbox_inches='tight', pad_inches=0)  # Adjust dpi as desired
        
        # Resize the image to 512x256 pixels
        image2 = Image.open(image)
        resized_image = image2.resize((640, 320))
        resized_image.save(image)
        # image = Image.open(image)

        print(i)

elif state_load_image == 'unnecessary':
    print("images have been saved once (table data have been saved as images)")


#%% Image to tensor
state = 'unnecessary' #load | save | unnecessary
if state == 'save':
    # Define the number of files and the file name pattern
    num_files = dt_in_T
    file_pattern_u = "u%d.png"
    
    # Initialize an empty list to store the images
    images_u = []
    
    # Loop through each file index and load the corresponding image
    for i in range(num_files):
        # Construct the file name using the file pattern
        file_name_u = file_pattern_u % i
        

        # Load the image using imageio
        image = imageio.imread(file_name_u)
        images_u.append(image)
    
    # Convert the list of images to a NumPy array
    U = np.array(images_u)
        
    #reorder for the CNN
    U = U.transpose(0, 3, 1, 2)

    #remove the last (transparency) channel
    U = U[:,:3,:,:]
    
    np.save("S2_3ChannelImage.npy", (U))
        
            

elif state == 'load':
    loaded_variables = np.load("S2_3ChannelImage.npy", allow_pickle=True)
    U = loaded_variables
    del loaded_variables
    #remove PNG channel    
    U = U[:,:3,:,:]
    
    print('3ChannelImage is loaded based on cropped images')

elif state == 'unnecessary':
    print("there is no need for 3channel image. Next I load 1channel (greyscale)")


#%% greying images
state = 'unnecessary' #load | save | unnecessary
if state == 'save':
    ##pixel data, #height pixel & width pixel
    h, w = U.shape[2], U.shape[3]
    
    #grey scaling
    U_temp = np.transpose(U, (0, 2, 3, 1))
    U_grey = np.dot(U_temp[..., :3], [0.2989, 0.5870, 0.1140])
    
    print('this is test of grey scaling:')
    #grey scaled image
    image = U_grey[100,:,:]
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Optional: to remove the axes and ticks
    plt.show()
    
    #just to validate with U before greyscaling
    image = np.transpose(U[100,:,:,:], (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')  # Optional: to remove the axes and ticks
    plt.show()
    
    U_grey = U_grey.reshape(dt_in_T, 1, h, w)
    U = U_grey
    np.save("S3_1channelImage.npy", (U))

elif state == 'load':
    U = np.load("S3_1channelImage.npy", allow_pickle=True)
    print('1channel Image (unscaled) is loaded based on cropped images')
    
elif state == 'unnecessary':
    print('Scalled 1channel image will be read directly in the next part')
































#%% Scaling
state = 'unnecessary' #load | save | unnecessary
if state == 'save':

    scaler_U = MinMaxScaler()
    min_U = 0
    max_U = 255
    
    U = U.reshape(-1,1)
    U = scaler_U.fit_transform(U)
    U = U.reshape(dt_in_T,1,h,w)
    np.save("S4_1channelImageScaled.npy", (U))
    
elif state == 'load':
    U = np.load("S4_1channelImageScaled.npy", allow_pickle=True)
    
elif state == 'unnecessary':
    print("I wanna load directly the torch tensor, which contatins also  the rotated images")

#%% CNN Class
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=9, out_features=9, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class Encoder(nn.Module):
    def __init__(self, input_shape=[128, 128, 1], num_clusters=9, filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.bn4_1 = nn.BatchNorm2d(filters[3])
        self.conv5 = nn.Conv2d(filters[3], filters[4], 5, stride=2, padding=2, bias=bias)
        self.embedding = nn.Linear(102400, num_clusters, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        x = self.sig(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        clustering_out = self.clustering(x)
        return x, clustering_out


class Decoder(nn.Module):
    def __init__(self, input_shape=[128, 128, 1], num_clusters=9, filters=[32, 64, 128, 256, 512], bias=True):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.deembedding = nn.Linear(num_clusters, 102400, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=out_pad, bias=bias)
        self.bn5_2 = nn.BatchNorm2d(filters[3])
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.deembedding(x)
        x = self.relu(x)
        x = x.view(x.size(0), 512, 10, 20)
        x = self.deconv5(x)
        x = self.relu(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        x = self.tanh(x)
        return x



class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        # Preprocess or transform the sample if needed
        return sample

    def __len__(self):
        return len(self.data)

#%% data preparation
state = 'load' #load | save
if state == 'save':
    U1 = torch.from_numpy(U).float()
    U2 = torch.flip(U1, [3])
    U_torch = torch.cat((U1, U2), dim=0)
    
    torch.save(U_torch, 'S5_UFlipped.pt')

elif state == 'load':
    U_torch = torch.load('S5_UFlipped.pt')

print("Now I have torch tensor of 2*n_snapshots contatins the original scalled 1channeld image and its flipped")
    
num_epochs = 500
in_dim = 28
lr = 0.01

dataset = CustomDataset(U_torch)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder = Encoder()
decoder = Decoder()

criterion = nn.MSELoss()
optimizer_fn = optim.Adam
optimizer = optimizer_fn(encoder.parameters(), lr=lr)
optimizer = optimizer_fn(decoder.parameters(), lr=lr)

#%% train CNN

CNN_mode = 'load' # load | train

if CNN_mode == 'train':
    
    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        tloss = 0.0
        with tqdm(data_loader, unit='batch') as progress_bar:
            for data in progress_bar:
                inputs = data
                optimizer.zero_grad()
                encoded, clustering_out = encoder(inputs)
                decoded = decoder(encoded)      
                loss = criterion(decoded, inputs)
                loss.backward()
                optimizer.step()
                tloss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
        epoch_loss = tloss / len(U_torch)
        torch.save(encoder.state_dict(), 'encoder.pth')
        torch.save(decoder.state_dict(), 'decoder.pth')
        print('Epoch loss: {:4f}'.format(epoch_loss))

elif CNN_mode == 'load':
    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

i=750
test_image = U_torch[i:i+1,:,:,:]
encoded, clustering_out = encoder(test_image)
decoded = decoder(encoded) 

torchvision.utils.save_image(test_image.data, 'in.png')
torchvision.utils.save_image(decoded.data, 'out.png')


#%%#############################LSTM##########################
#%% Latent Space Generation

# remove the flipped part
U = np.load("S4_1channelImageScaled.npy", allow_pickle=True)
U = torch.from_numpy(U).float()

#generating latent space for all snapshots
encoder.eval()
with torch.no_grad():
  for i in range(len(U)):
    q = torch.tensor(U[i:i+1, :, :, :], dtype=torch.float)  #(1, 3, ?, ?)

    out, clustering_out = encoder(q)

    if i == 0:
      encoded = out
    else:
      encoded = torch.concat([encoded, out], dim=0)

#%% Add BC to latent space matrix
t = np.linspace(1, n_snapshots, num=n_snapshots)
inputt = np.concatenate((t.reshape(-1, 1), interpolated_v_values.reshape(-1, 1), interpolated_P_values.reshape(-1, 1), encoded.numpy()), axis=1)
outputt = encoded.numpy()

inputt = inputt.T
outputt = outputt.T

#%% Latent Space Scaling

min_vals_input = np.min(inputt, axis=1) 
max_vals_input = np.max(inputt, axis=1) 
input_scaled = inputt - min_vals_input[:, np.newaxis]
input_scaled /= (max_vals_input - min_vals_input)[:, np.newaxis]

min_vals_output = np.min(outputt, axis=1) 
max_vals_output = np.max(outputt, axis=1) 
output_scaled = outputt - min_vals_output[:, np.newaxis]
output_scaled /= (max_vals_output - min_vals_output)[:, np.newaxis]

#%% Time sequencing

input_scaled = input_scaled.T
output_scaled = output_scaled.T

time_window = 100

x_train = []
y_train = []

for i in range(0,len(input_scaled) - time_window -1): 
    x_train.append( input_scaled[i : (i+time_window) , :] ) 
    y_train.append( output_scaled[i+time_window,:]) 
    
total_x = np.array(x_train)
total_y = np.array(y_train)

#%% Splitting train and test
Test_split = 0.8

x_train = total_x[:int(Test_split * total_x.shape[0]),:,:]
y_train = total_y[:int(Test_split * total_y.shape[0])]

x_test = total_x[int(Test_split * total_x.shape[0]):, :,:]
y_test = total_y[int(Test_split * total_y.shape[0]):]

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


#%% Class LSTM
input_size = 12
hidden_size = 512
output_size = 9
sequence_length = time_window
learning_rate = 0.000005
num_epochs = 2000
batch_size = 20
early_stop_patience = 100
dropout_prob = 0.2

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 9)

        # Initialize LSTM weights
        for name, param in self.lstm1.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

        # Initialize linear layer weights
        init.xavier_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        h01 = torch.zeros(3, x.size(0), self.hidden_size).requires_grad_()
        c01 = torch.zeros(3, x.size(0), self.hidden_size).requires_grad_()
        out1, (h01, c01) = self.lstm1(x, (h01.detach(), c01.detach()))
        out = nn.functional.relu(self.fc1(out1[:, -1, :]))
        return out

model_lstm = LSTMNet(input_size, hidden_size, output_size, dropout_prob)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)
    
#%% Train LSTM
best_loss = float('inf')
early_stop_count = 0

state_lstm = 'train'  # train | load

# load the model
checkpoint = torch.load('my_lstm_model.pt')
model_lstm.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

if state_lstm == 'train': 
    for epoch in range(num_epochs):
        model_lstm.train()
        for i in range(0, x_train.shape[0], batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
    
            # Forward pass
            outputs = model_lstm(batch_x)
    
            # Compute loss and backpropagation
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Evaluate the model on training and test data
        with torch.no_grad():
            model_lstm.eval()
            train_outputs = model_lstm(x_train)
            train_loss = criterion(train_outputs, y_train)
            test_outputs = model_lstm(x_test)
            test_loss = criterion(test_outputs, y_test)
    
        # Print training and test loss every epoch
        print('Epoch [{}/{}], Train Loss: {:.6f}, Test Loss: {:.6f}'.format(epoch+1, num_epochs, train_loss.item(), test_loss.item()))
    
        # Check if the current test loss is the best so far
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_count = 0
          
            # Save your model and optimizer
            torch.save({
                'model_state_dict': model_lstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': num_epochs
            }, 'my_lstm_model.pt')
    
            
        else:
            early_stop_count += 1
            if early_stop_count == early_stop_patience:
                print('Early stopping after {} epochs'.format(epoch+1))
                break

elif state_lstm == 'load':
    # load the model
    checkpoint = torch.load('my_lstm_model.pt')
    model_lstm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']


#%% Prediction with predicted data (Nu predicted is used for each prediction)
with torch.no_grad():
    x = x_test[0:1,:,:]
    x = torch.tensor(x, dtype=torch.float32)
    model_lstm.eval()
    predict = model_lstm(x)
    pred_final = []
    
    for i in range (1,x_test.shape[0]):
        pred_final.append(predict)
        x = x_test[i:i+1, :, :]
        x = torch.tensor(x, dtype=torch.float32)
        # x[-1,-1,-5:] = predict
        predict = model_lstm(x)

pred_final = torch.cat(pred_final, dim=0).detach().numpy()
    

# Create a plot
for i in range(10):
    plt.plot(pred_final[:,i], label='NN Prediction')
    plt.plot(y_test[:,i], label='Actual Data')
    plt.legend()
    plt.show()

#%% Decoding the Latent of prediction

# actual latent space for all snapshots
latent_actual = inputt

# remove first three rows of 'actual' are time, v_in and P_out 
latent_actual = latent_actual[3:,:]


#prediction data (scaled) (for all snapshots)
pred_scaled = np.concatenate( (y_train.detach().numpy(), pred_final), axis=0)
#prediction data (unscaled)
pred_unscaled = pred_scaled.T * (max_vals_output - min_vals_output)[:, np.newaxis] + min_vals_output[:, np.newaxis]

#ROM results start from time_window to the end. I want to concat the first time_window values to the start of ROM. It this way, both pred and actual start from same values
latent_predict = np.concatenate( (latent_actual[:,:100],pred_unscaled), axis=1 ) #[latent size,snapshot-2]


#change type to torch
latent_actual = torch.tensor(latent_actual, dtype=torch.float)
latent_predict = torch.tensor(latent_predict, dtype=torch.float)

#decode
actual = decoder(latent_actual[:,-500:].T)
predict = decoder(latent_predict[0:,-500:].T)

#%% Test prediction with image
i=-1 #end time
torchvision.utils.save_image(actual[i].data, 'CNNlstm_actual.png')
torchvision.utils.save_image(predict[i].data, 'CNNlstm_predict.png')


image = imageio.imread('CNNlstm_actual.png')
image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
plt.imshow(image)
plt.axis('off')  # Optional: to remove the axes and ticks
plt.show()

image = imageio.imread('CNNlstm_predict.png')
image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
plt.imshow(image)
plt.axis('off')  # Optional: to remove the axes and ticks
plt.show()


#%% L2 norm error

# l2_norm_diff = np.linalg.norm(actual.detach().numpy() - predict.detach().numpy(), axis=(2, 3))

# plt.plot(l2_norm_diff)
# plt.xlabel('Snapshot')
# plt.ylabel('L2 Norm Difference')
# plt.title('L2 Norm Difference between Actual and Predicted Snapshots')
# plt.show()

actual_ = actual.reshape(len(actual),-1).detach().numpy()
predict_ = predict.reshape(len(predict),-1).detach().numpy()

RelErr1 = np.zeros((len(actual_)))
for t in range(len(actual)):
        # print(t)
        RelErr1[t] = (norm(actual_[t,:] - predict_[t,:])/norm(actual_[t,:]) ) * 100

plt.plot(RelErr1)
plt.xlabel('Snapshot')
plt.ylabel('L2 Norm Difference')
plt.title('L2 Norm Difference between Actual and Predicted Snapshots')
plt.show()










    
    
