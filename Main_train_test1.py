import torch
import scipy.io as scio

import torch.optim as optim
from RIS_Net_define1 import PhaseNN
from torch.autograd import Variable

device = torch.device("cpu")

# system parameters ------------------------------------------
M = 4      # the number of BS antennas
N = 100    # the number of RIS elements

# generate the input data for training ------------------------
# obtain the path
mat_file_path = "../DLStudy/channel_Data.mat"
# read MAT file
mat_data = scio.loadmat(mat_file_path)

G_channel    = torch.tensor(mat_data['G'])
hr_channel   = torch.tensor(mat_data['Hr'])
G_channel    = G_channel.permute(2, 1, 0)             # Dimension: [batch, M, N]
hr_channel   = hr_channel.permute(2, 1, 0)[:, :, 1]   # Dimension: [batch, N]
hr_channel_diag   = torch.diag_embed(hr_channel)      # Dimension: [batch, N]

G_channel = torch.tensor(G_channel, dtype=torch.complex64)
hr_channel = torch.tensor(hr_channel, dtype=torch.complex64)
hr_channel_diag = torch.tensor(hr_channel_diag, dtype=torch.complex64)

# Generate feature vector
F_matrix = torch.matmul(G_channel, hr_channel_diag)   # F_vector = G_channel * hr_channel
F_matrix_real = F_matrix.real
F_matrix_imag = F_matrix.imag
F_matrix_input = torch.cat([F_matrix_real.unsqueeze(0), F_matrix_imag.unsqueeze(0)], dim=0)
F_matrix_input = F_matrix_input.permute(1, 0, 2, 3)    # Dimension: [batch, 2, M, N]


# Training function
def train_epoch(model, optimizer, F_matrix_input, G_channel, hr_channel):

    model.train()
    Loss = model(F_matrix_input.float(), G_channel, hr_channel).to(device)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    return Loss


if __name__ == "__main__":

    #  initialize the optimizer and train model
    Train_model  = PhaseNN().to(device)
    optimizer    = optim.Adam(Train_model.parameters(), lr=0.001)

    for i in range(500):
        print("Epoch is ", i)
        train_loss = train_epoch(Train_model, optimizer, F_matrix_input, G_channel, hr_channel)
        print("train_loss = ", train_loss)


