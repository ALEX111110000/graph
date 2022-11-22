#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ReLU(), #*
        nn.Dropout(p=p_drop),
    )


# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj #*返回A‘


class SEDR(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        
        '''
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))  #*
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop)) #*feat_hidden2才是feat_x维度。

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop)) #*是X‘解码器
        '''    
        self.fc1 = nn.Linear(784, 400, bias = False) # Encoder
		self.fc2 = nn.Linear(400, 784, bias = False) # Decoder

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
        
        # GCN layers
        self.gc1 = GraphConvolution(params.feat_hidden2, params.gcn_hidden1, params.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x) #*self.dc是A'解码器，VGAE。输入z，返回adj。 z~gnn_z~self.reparameterize(mu, logvar)~return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x
                                                                                                                                                    #*                  mu, logvar, feat_x = self.encode(x, adj)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
'''
    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x
'''
	def encode(self, x, adj):
		feat_x = self.relu(self.fc1(x.view(-1, 784)))
		return feat_x   #*h1变成feat_x
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

	def decoder(self,z):
		de_feat = self.sigmoid(self.fc2(z))
		return de_feat
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class Discriminator(Model):
    def __init__(self, inputdim, dc_hidden1_dim, dc_hidden2_dim, **kwargs):  #注意input_dim和Deeplinc函数的hidden2_dim要一样
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu
        self.input_dim = inputdim
        self.dc_h1_dim = dc_hidden1_dim
        self.dc_h2_dim = dc_hidden2_dim

    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, self.input_dim, self.dc_h1_dim, name='dc_den1'))  #125,150
            dc_den2 = tf.nn.relu(dense(dc_den1, self.dc_h1_dim, self.dc_h2_dim, name='dc_den2'))  #150,125
            output = dense(dc_den2, self.dc_h2_dim, 1, name='dc_output')
            return output


