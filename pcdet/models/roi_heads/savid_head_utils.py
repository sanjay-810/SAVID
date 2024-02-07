import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1,2)

        return x

class CPConvs(nn.Module):
    def __init__(self):
        super(CPConvs, self).__init__()
        self.pointnet1_fea = PointNet(  6,12)
        self.pointnet1_wgt = PointNet(  6,12)
        self.pointnet1_fus = PointNet(108,12)

        self.pointnet2_fea = PointNet( 12,24)
        self.pointnet2_wgt = PointNet(  6,24)
        self.pointnet2_fus = PointNet(216,24)

        self.pointnet3_fea = PointNet( 24,48)
        self.pointnet3_wgt = PointNet(  6,48)
        self.pointnet3_fus = PointNet(432,48)

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features

        N, F = points_features.shape
        N, M = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()
        points_neighbor[point_empty[:,0], point_empty[:,1]] = point_empty[:,0]

        pointnet_in_xiyiziuiviri = torch.index_select(points_features[:,[0,1,2,6,7,8]],0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet_in_x0y0z0u0v0r0 = points_features[:,[0,1,2,6,7,8]].unsqueeze(dim=1).repeat([1,M,1])
        pointnet_in_xyzuvr       = pointnet_in_xiyiziuiviri - pointnet_in_x0y0z0u0v0r0
        points_features[:, 3:6] /= 255.0
        
        pointnet1_in_fea        = points_features[:,:6].view(N,1,-1)
        pointnet1_out_fea       = self.pointnet1_fea(pointnet1_in_fea).view(N,-1)
        pointnet1_out_fea       = torch.index_select(pointnet1_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet1_out_wgt       = self.pointnet1_wgt(pointnet_in_xyzuvr)
        pointnet1_feas          = pointnet1_out_fea * pointnet1_out_wgt
        pointnet1_feas          = self.pointnet1_fus(pointnet1_feas.reshape(N,1,-1)).view(N,-1)   

        pointnet2_in_fea        = pointnet1_feas.view(N,1,-1)
        pointnet2_out_fea       = self.pointnet2_fea(pointnet2_in_fea).view(N,-1)
        pointnet2_out_fea       = torch.index_select(pointnet2_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet2_out_wgt       = self.pointnet2_wgt(pointnet_in_xyzuvr)
        pointnet2_feas           = pointnet2_out_fea * pointnet2_out_wgt
        pointnet2_feas          = self.pointnet2_fus(pointnet2_feas.reshape(N,1,-1)).view(N,-1)

        pointnet3_in_fea        = pointnet2_feas.view(N,1,-1)
        pointnet3_out_fea       = self.pointnet3_fea(pointnet3_in_fea).view(N,-1)
        pointnet3_out_fea       = torch.index_select(pointnet3_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet3_out_wgt       = self.pointnet3_wgt(pointnet_in_xyzuvr)
        pointnet3_feas           = pointnet3_out_fea * pointnet3_out_wgt
        pointnet3_feas          = self.pointnet3_fus(pointnet3_feas.reshape(N,1,-1)).view(N,-1)
 
        pointnet_feas     = torch.cat([pointnet3_feas, pointnet2_feas, pointnet1_feas, points_features[:,:6]], dim=-1)
        return pointnet_feas

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim

        for hidden_layer_size in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_layer_size))
            layers.append(activation_fn)
            last_dim = hidden_layer_size
        layers.append(nn.Linear(last_dim, output_dim))k
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class local(nn.Module):
    def __init__(self, image_dim, output_features_dim):
        super(local, self).__init__()
        self.mlp = MLP(image_dim, 3, image_dim)
        self.query_layer = nn.Linear(image_dim, output_features_dim)
        self.key_layer = nn.Linear(image_dim, output_features_dim)
        self.value_layer = nn.Linear(image_dim, output_features_dim)

    def forward(self,image):
        query = self.query_layer(self.mlp(image))
        key = self.key_layer(self.mlp(image))
        value = self.value_layer(self.mlp(image))
        query = nn.LayerNorm(query)
        key = nn.LayerNorm(key)
        value = nn.LayerNorm(value)
        attention_scores = torch.matmul(query, key.transpose(2, 1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs= torch.fft(attention_probs)
        attended_features = torch.matmul(attention_probs, value)
        attended_features= torch.ifft(attended_features)
        return attended_features
    
class GMAN(nn.Module):
    def __init__(self, image_dim, depth_dim, d_model, num_heads):
        super(GMAN, self).__init__() 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.lmsa = local(image_dim,image_dim)
        self.mlp = MLP(image_dim, 3, image_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)       
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    def forward(self, depth, image, mask=None):
        image = self.lmsa(image)
        query = self.query_layer(self.mlp(depth))
        key = self.key_layer(self.mlp(image))
        value = self.value_layer(self.mlp(image))
        batch_size = query.size(0)
        query = nn.LayerNorm(query)
        key = nn.LayerNorm(key)
        value = nn.LayerNorm(value)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output_1 = torch.matmul(attention_weights, V)
        output_1 = output_1.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        lstm = nn.LSTM(output_1.shape)
        return self.fc(lstm(output_1,output_1.shape))
    
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(1, hidden_size))
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(1, hidden_size))
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.bc = nn.Parameter(torch.zeros(1, hidden_size))
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.bo = nn.Parameter(torch.zeros(1, hidden_size))

    def forward(self, lidar,image, states):
        query = self.query_layer(lidar)
        key = self.key_layer(image)
        value = self.value_layer(image)
        ash = torch.matmul(query, key.transpose(2, 1))
        x=nn.BatchNorm3d(ash)
        h_prev, C_prev = states
        h_prev = nn.BatchNorm3d(h_prev)
        combined = torch.cat([h_prev, x], dim=1)
        ft = torch.relu(self.Wf(combined) + self.bf)
        it = torch.relu(self.Wi(combined) + self.bi)
        C_tilda = torch.tanh(self.Wc(combined) + self.bc)
        it = torch.matmul(it, value.transpose(2, 1))
        C_tilda = torch.matmul(C_tilda, value.transpose(2, 1))
        Ct = ft * C_prev + it * C_tilda
        ot = torch.sigmoid(self.Wo(combined) + self.bo)
        ht = ot * torch.tanh(Ct)

        return ht, Ct

%%%%%% LidarFusion is the KGF function in the main paper


class LidarFusion(nn.Module):
    def __init__(self, sparse_shape, lidar_shape):
        super(LidarFusion, self).__init__()
        self.sparse_shape = sparse_shape
        self.lidar_shape = lidar_shape

    def cosine_distance(self, a, b):
        return (a*b) / np.sqrt((a**2) + (b**2))

    def Lidar_weight(self, pixel_value, Lidar, x, y):
        channels = Lidar.shape[0]
        x_range, y_range = Lidar[0].shape
        count = 0
        for i in range(channels):
            temp = [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]
            ite = []
            for val in temp:
                if 0 <= val[0] <= x_range and 0 <= val[1] <= y_range:
                    ite.append(val)
            minn = np.float("inf")
            for a, b in ite:
                temp = self.cosine_distance(pixel_value, Lidar[i][a][b])
                if temp < minn:
                    minn, x, y = temp, a, b
            count += (2**(-i-1)) * minn
        return count

    def forward(self, sparse, Lidar):
        channels = sparse.shape[-1]
        sparse = sparse.transpose(2, 0, 1)
        Lidar = Lidar.transpose(2, 0, 1)
        output = np.zeros(sparse.shape)
        for i in range(channels):
            x_len, y_len = sparse[i].shape
            for x in range(x_len):
                for y in range(y_len):
                    pixel_value = sparse[i][x][y]
                    lidar_value = self.Lidar_weight(pixel_value, Lidar[i:], x, y)
                    output[i][x][y] = pixel_value + lidar_value
        return output



