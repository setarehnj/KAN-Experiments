import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


# T_n(x) = cos (n * arccos(x)) for x in [-1, 1]

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.randn(input_dim, output_dim, degree + 1) / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(degree + 1).view(1, 1, -1))

    def forward(self, x):

        x = x.view(-1, self.input_dim)
        x = torch.tanh(x)
        #print(f"Input to ChebyKANLayer: {x.shape}")
        x = x.unsqueeze(-1)  # Shape: (batch_size, input_dim, 1)
        #print(f"arange tensor shape: {self.arange.shape}")
        
        # Compute Chebyshev polynomials efficiently
        cheby_polys = torch.cos(self.arange * torch.acos(x))
        #print(f"Output from ChebyKANLayer after cos: {cheby_polys.shape}")
        #print(f"Cheby coeffs shape: {self.cheby_coeffs.shape}")
        # Use batch matrix multiplication
        y = torch.einsum('bid,iod->bo', cheby_polys, self.cheby_coeffs)


        #print(f"Output from ChebyKANLayer after einsum: {y.shape}")

        
        return y.view(1, -1, 1)



class ChebyshevNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, cheby_degree):
        super(ChebyshevNetwork, self).__init__()
        layers = []

        # Adding the first ChebyKANLayer
        layers.append(('cheby_1', ChebyKANLayer(input_dim, hidden_dim, cheby_degree)))

        # Adding subsequent ChebyKANLayer layers
        for i in range(num_layers - 1):
            layers.append((f'cheby_{i+2}', ChebyKANLayer(hidden_dim, hidden_dim, cheby_degree)))

        # Adding the final ChebyKANLayer which outputs to the required output_dim
        layers.append((f'cheby_final', ChebyKANLayer(hidden_dim, output_dim, cheby_degree)))

        # Creating the model as an ordered dictionary
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class SingleBVPNetwithKAN(nn.Module):
    '''A canonical representation network for a BVP using Chebyshev networks.'''

    def __init__(self, out_features=1, in_features=2, hidden_features=256, num_hidden_layers=3, chebyshev_degree=5, **kwargs):
        super().__init__()
        self.net = ChebyshevNetwork(input_dim=in_features, hidden_dim=hidden_features, num_layers=num_hidden_layers,
                                    output_dim=out_features, cheby_degree=chebyshev_degree)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
class FourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_terms, learnable_frequencies=False):
        super(FourierKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_terms = num_terms
        self.learnable_frequencies = learnable_frequencies
        
        # Learnable fundamental frequency (gridsize)
        self.gridsize_param = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        
        # Learnable frequency factors (if enabled)
        if learnable_frequencies:
            self.frequency_factors = nn.Parameter(torch.randn(input_dim, num_terms))
        else:
            self.register_buffer('frequency_factors', torch.arange(1, num_terms + 1).unsqueeze(0).expand(input_dim, -1).float())
        
        # Learnable Fourier coefficients
        # self.fouriercoeffs = nn.Parameter(torch.empty(input_dim, output_dim, 2, num_terms))
        # nn.init.xavier_uniform_(self.fouriercoeffs)

        self.fouriercoeffs = nn.Parameter(torch.empty(input_dim, output_dim, 2, num_terms))
        nn.init.normal_(self.fouriercoeffs, mean=0.0, std=1 / (input_dim * num_terms * 2))

    def forward(self, x):
        #print(f"Input to FourierKANLayer: {x.shape}")

        x = x.view(-1, self.input_dim, 1)
        #print(f"Reshaped input to FourierKANLayer: {x.shape}")
            
        # Compute the Fourier series terms
        frequencies = self.frequency_factors / self.gridsize_param
        #print(f"Frequencies shape: {frequencies.shape}")
        
        angles = 2 * math.pi * frequencies.unsqueeze(0) * x
        #print(f"Angles shape: {angles.shape}")
        
        # Compute cosine and sine terms
        fourier_terms = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-2)
        #print(f"Fourier terms shape: {fourier_terms.shape}")
        
        # Multiply with Fourier coefficients and sum
        output = torch.einsum('bifd,iofd->bo', fourier_terms, self.fouriercoeffs)
        output = output.view(1, -1, 1)
            
        #print(f"Output from FourierKANLayer: {output.shape}")
        return output


class FourierKANNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_terms, learnable_frequencies=True):
        super(FourierKANNetwork, self).__init__()
        layers = []

        # Adding the first FourierKANLayer
        layers.append(('fourier_1', FourierKANLayer(input_dim, hidden_dim, num_terms, learnable_frequencies)))

        # Adding subsequent FourierKANLayer layers
        for i in range(num_layers - 1):
            layers.append((f'fourier_{i+2}', FourierKANLayer(hidden_dim, hidden_dim, num_terms, learnable_frequencies)))

        # Adding the final FourierKANLayer which outputs to the required output_dim
        layers.append((f'fourier_final', FourierKANLayer(hidden_dim, output_dim, num_terms, learnable_frequencies)))

        # Creating the model as an ordered dictionary
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)

class SingleBVPNetwithFourier(nn.Module):
    '''A canonical representation network for a BVP using Fourier KAN networks.'''

    def __init__(self, out_features=1, in_features=2, hidden_features=256, num_hidden_layers=3, num_terms=6, learnable_frequencies=False, **kwargs):
        super().__init__()
        self.net = FourierKANNetwork(
            input_dim=in_features, 
            hidden_dim=hidden_features, 
            num_layers=num_hidden_layers,
            output_dim=out_features, 
            num_terms=num_terms,
            learnable_frequencies=learnable_frequencies
        )
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output} 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class WaveletKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, initial_gridsize=10):
        super(WaveletKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))
        
        # Wavelet coefficients as a learnable parameter
        # self.waveletcoeffs = nn.Parameter(torch.empty(input_dim, output_dim, 2, initial_gridsize))
        # nn.init.xavier_uniform_(self.waveletcoeffs)

        self.waveletcoeffs = nn.Parameter(torch.empty(input_dim, output_dim, 2, initial_gridsize))
        nn.init.normal_(self.waveletcoeffs, mean=0.0, std=1 / (input_dim * initial_gridsize * 2))
        
    def forward(self, x):
        #print(f"Input to WaveletKANLayer: {x.shape}")
        
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        
        # Reshape input for processing
        x = x.view(-1, self.input_dim, 1)
        #print(f"Reshaped input to WaveletKANLayer: {x.shape}")
        
        scales = torch.linspace(1, gridsize, gridsize, device=x.device).view(1, 1, -1)
        translations = torch.linspace(0, 1, gridsize, device=x.device).view(1, 1, -1)
        
        u = (x - translations) * scales
        #print(f"u shape: {u.shape}")
        
        # Compute wavelet terms
        wavelet_terms = torch.stack([
            torch.cos(math.pi * u) * torch.exp(-u**2 / 2.),
            torch.sin(math.pi * u) * torch.exp(-u**2 / 2.)
        ], dim=-2)
        #print(f"Wavelet terms shape: {wavelet_terms.shape}")
        
        # Multiply with wavelet coefficients and sum
        y = torch.einsum('bifd,iofd->bo', wavelet_terms, self.waveletcoeffs[..., :gridsize])
        y = y.view(1, -1, 1)
        
        #print(f"Output from WaveletKANLayer: {y.shape}")
        return y

class WaveletKANNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, initial_gridsize=10):
        super(WaveletKANNetwork, self).__init__()
        layers = []

        # Adding the first WaveletKANLayer
        layers.append(('wavelet_1', WaveletKANLayer(input_dim, hidden_dim, initial_gridsize)))

        # Adding subsequent WaveletKANLayer layers
        for i in range(num_layers - 1):
            layers.append((f'wavelet_{i+2}', WaveletKANLayer(hidden_dim, hidden_dim, initial_gridsize)))

        # Adding the final WaveletKANLayer which outputs to the required output_dim
        layers.append((f'wavelet_final', WaveletKANLayer(hidden_dim, output_dim, initial_gridsize)))

        # Creating the model as an ordered dictionary
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)

class SingleBVPNetwithWavelet(nn.Module):
    '''A canonical representation network for a BVP using Wavelet KAN networks.'''

    def __init__(self, out_features=1, in_features=2, hidden_features=256, num_hidden_layers=3, initial_gridsize=10, **kwargs):
        super().__init__()
        self.net = WaveletKANNetwork(
            input_dim=in_features, 
            hidden_dim=hidden_features, 
            num_layers=num_hidden_layers,
            output_dim=out_features, 
            initial_gridsize=initial_gridsize
        )
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}
#------------------------------------------------------------------------------------------
class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        #print(f"Input to BatchLinear: {input.shape}")
        
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        #print(f'DNN Output shape: {output.shape}') 
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output


class SingleBVPNetwithDNN(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def init_weights_trunc_normal(m):
    if type(m) == ChebyKANLayer or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)

def init_weights_normal(m):
    if type(m) == ChebyKANLayer or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_selu(m):
    if type(m) == ChebyKANLayer or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == ChebyKANLayer or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def init_weights_xavier(m):
    if type(m) == ChebyKANLayer or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)



