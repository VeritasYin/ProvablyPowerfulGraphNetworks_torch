import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules
import pdb


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()
        self.bins  = 8
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.is_SimGNN = config.dataset_name in ['LINUX', 'AIDS700nef']
        self.use_NTN = True
        self.use_histogram = True
        use_new_suffix = config.architecture.new_suffix  # True or False
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            if config.simple:
                mlp_block = modules.MlpBlock(last_layer_features, next_layer_features, config.architecture.depth_of_mlp)
            else:
                mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2*output_features, self.config.num_classes, activation_fn=None)
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))
            
        # SimGNN layers (graph rep -> ged)
        if self.is_SimGNN:
            if self.use_NTN:
                self.SimGNN_NTN = modules.NTN(self.config.num_classes, 256)
                first_layer_features = 256
            else:
                first_layer_features = self.config.num_classes*2
            if self.use_histogram:
                first_layer_features += self.bins*2
            self.SimGNN_layers = nn.ModuleList()
            self.SimGNN_layers.append(modules.FullyConnected(first_layer_features, 256))
            self.SimGNN_layers.append(modules.FullyConnected(256, 128))
            self.SimGNN_layers.append(modules.FullyConnected(128, 1, activation_fn=None))
            
    def calculate_histogram(self, matf1, matf2):
        """
        Calculate histogram from similarity matrix.
        :param matf1: Feature matrix for type 1.
        :param matf2: Feature matrix for type 2.
        :return hist: Histsogram of similarity scores.
        """
        bs, F, M = matf1.shape
        nbs = bs**2
        hist = torch.empty(nbs, self.bins*2, device=self.device)
        input1 = matf1.expand(bs,bs,F,M)
        input2 = matf2.expand(bs,bs,F,M)
        scores1 = torch.matmul(torch.transpose(torch.transpose(input1, 0, 1), -2, -1), input1).view(nbs, -1).detach()
        scores2 = torch.matmul(torch.transpose(torch.transpose(input2, 0, 1), -2, -1), input2).view(nbs, -1).detach()
        for i in range(nbs):
            hist[i] = torch.cat((nn.functional.normalize(torch.histc(scores1[i], bins=self.bins), p=2, dim=0),nn.functional.normalize(torch.histc(scores2[i], bins=self.bins), p=2, dim=0)), dim = -1)
        return hist

    def forward(self, input):
        x = input
        scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

            if self.config.architecture.new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix
            bs = x.shape[0]
            x_max = torch.max(x, dim=2)[0]
            x_sum = torch.sum(x, dim=2)
            if self.use_histogram:
                hist = self.calculate_histogram(x_max, x_sum)
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        # If SimGNN, pair up and apply anothe 3-layer MLP
        # Afterwards scores has shape (n,n)
        if self.is_SimGNN:
            n = scores.shape[0]
            if self.use_NTN:
                x = self.SimGNN_NTN(scores)
            else:
                d = scores.shape[1]
                input1 = scores.expand(n,n,d)
                input2 = torch.transpose(input1,0,1)
                x = torch.reshape(torch.cat((input1,input2), 2) , (n*n,2*d))
            if self.use_histogram:
                x = torch.cat((x, hist), -1)
            for fc in self.SimGNN_layers:
                x = fc(x)
            geds = torch.reshape(x, (n,n))

        return geds
