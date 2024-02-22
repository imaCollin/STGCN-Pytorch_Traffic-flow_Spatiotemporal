# STGNN-Pytorch_Traffic-flow_Spatiotemporal
## Characteristics:
The proposed model represents a pioneering approach in traffic research by employing pure convolutional layers (TCN) to simultaneously extract spatial-temporal information from time series of graph structures. It introduces a novel neural network architecture comprising spatio-temporal blocks. Due to the exclusive use of convolution operations, this architecture exhibits training speeds over 10 times faster than RNN-based models and requires fewer parameters. Moreover, it enables more efficient handling of larger road networks. Experimental results validate the effectiveness of the proposed network on two real traffic datasets.
## Network Skeleton:
The STGCN architecture consists of multiple space-time convolution blocks, each exhibiting a "sandwich" structure comprising two gate sequence convolution layers sandwiching a spatial graph convolution layer. In terms of its component structure, STGCN comprises two fast ST-Conv blocks (light blue section) and a fully connected output layer (green section), wherein each ST-Conv block encompasses two time convolution blocks (orange section) and one spatial convolution block (light yellow section).
![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/899eae87-d367-42e6-a53a-0d9f54165a5e)

_Figure of STGCN_

The Graph convolutional neural network that extracts time features is the Spatial Graph-conv module in the corresponding network structure. The graph convolutional neural network that extracts spatial features is the Temporal Gated-Conv module in the corresponding network structure. Hence, the ST-Conv Block formula provides an alternative interpretation of the figure. Initially, the input data undergoes convolution in the time dimension, followed by convolution in the graph dimension for the output result. Subsequently, a RELU activation is applied to the output of graph convolution, and finally, a convolution operation is performed in the time dimension to obtain the overall output of the ST-Conv Block.
The eventual model consists of two stacked St-Conv blocks followed by an output layer. The output layer combines the temporal dimension of the previous output data with the convolution of the temporal dimension, and then generates the final prediction data through a convolution.

### Step_1: Reading the data.utils.py
utlis.load_metr_la_data(): Read the data and normalize it using the Z-score method.

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/409bbb1f-a45d-4b7c-8582-3b05f3e5dabd)

utlis.get_normalized_adj(): The adjacency matrix is processed and the degree information is extracted.

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/acdcf379-f2cc-42a1-b950-6f59f009fbf8)

utlis.generate_dataset(): The output of generate_dataseth is X, which has dimensions (num_samples, num_vertices, num_features, num_timesteps_input), specifically [20549, 207, 12, 2]. Here, 20549 represents the number of processed time slices, 207 denotes the number of nodes present in the dataset. Additionally, there are 12 features corresponding to the last 12 time slices and each feature consists of two channels. On the other hand, Y obtained from generate_dataseth has dimensions (num_samples,num_vertices,num_features,num_timesteps_output) equal to [20549 ,207 ,2], where the first channel corresponds to the predicted values.

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/88f214fa-1d51-44b5-bae9-2195fc5d415f)

### Step_2: Dimension transformation:
_STGCN = block1+block2+last_temporal_
The first TimeBlock within the first Block is defined as self.temporal1TimeBlock(in_channels=in_channels,out_channels=out_channels), The TimeBlock input is [50, 207, 12, 2], which is [B,H,W,C], but it needs further transformation, and the result after transformation is [50, 2, 207, 12], which is [B,C,H,W], because in general, when graph network data is converted to grid data,H generally represents the number of grid nodes. W represents the number of features (the amount of historical data in the same time slice under spatio-temporal data). Another reason for scaling is to correspond with the convolution kernel, because the size of the convolution kernel is (1, kernel_size=3, corresponding to H,W respectively),C represents the channel, and B represents the size of the batch. Then the 1D convolution is performed on W,C, **which is obviously unreasonable.**

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/d70089ef-f742-45d9-a4b3-2e3479aff331)

The result of TimeBlock is [50, 64, 207, 10], which is [B,C,H,W] (in the case of calling TimeBlock once,W is reduced by 2). The result returned is [50, 207, 10, 64], or [B,H,W,C], which is in the same format as the previous input, because we will also call TimeBlock later.

GLU is defined as self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,spatial_channels) - spatial_channels=16. After matrix multiplication, the output of GLU's operation transformation is [50, 207, 10, 16], which is used as the input of the next TimeBlock.

Finally, the fully connected layer: self.fully = nn.Linear((num_timesteps_input-2 * 5) *64,num_timesteps_output), out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))), which yields [50, 207, 3]

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/f2742cb6-a2d7-4172-a784-8fb385072b2c)

![image](https://github.com/imaCollin/STGNN-Pytorch_Traffic-flow_Spatiotemporal/assets/127849702/ec85cc6d-a5c0-4a68-a390-0efff58f3dba)






