import numpy as np                                                                              # 01: core library
from skimage.util.shape import view_as_windows as unroll                                        # 02: wrapper to numpy's as_strided method for rolling window view

def pad(data, parameter):                                                                       # 03: PARAMETER as 8-element list for leading/trailing zeros in all 4 dimensions of DATA

    if np.count_nonzero(parameter) == 0: return data                                            # 04: early stop if nothing to be padded
    else: return np.pad(data, zip(parameter[0::2],parameter[1::2]), 'constant')                 # 05: zero padding

def forward(data, net):                                                                         # 06: DATA in format N*C*H*W (N*3*227*227 for REFNET input)

    for l in xrange(len(net)):                                                                  # 07: NET (list of layers)

        if net[l]['type'] == 'conv':                                                            # 08: CONVOLUTION or INNER_PRODUCT
            data = pad(data, [0,0,0,0]+net[l]['pad'])                                           # 09: padding DATA in spatial dimensions (H and W) if needed
            gnum = data.shape[1] / net[l]['filters'].shape[1]                                   # 10: number of FILTER groups (G)
            gdat = gnum * [None]                                                                # 11: empty list for result caching
            fdat = net[l]['filters'].reshape((gnum,-1)+net[l]['filters'].shape[1:])             # 12: reshaping FILTER (C'*C*h*w -> G*C'/G*C*h*w)
            for g in xrange(gnum):                                                              # 13: looping through FILTER groups
                gdat[g] = data.reshape((data.shape[0],gnum,-1) + data.shape[2:])[:,g]           # 14: reshaping DATA (N*C*H*W -> N*G*C/G*H*W) into groups and taking the corresponding group
                gdat[g] = unroll(gdat[g], (1,)+net[l]['filters'].shape[1:]).squeeze((1,4))      # 15: rolling window view of DATA (N*C*H*W -> N*(H-h+1)*(W-w+1)*C*h*w)
                gdat[g] = gdat[g][:,::net[l]['stride'][0],::net[l]['stride'][1]]                # 16: striding (before computation)
                gdat[g] = np.tensordot(gdat[g], fdat[g], axes=([3,4,5],[1,2,3]))                # 17: convolution (or inner product)
            data = np.concatenate(tuple(gdat), -1).transpose(0,3,1,2)                           # 18: concatenating results from all groups and transposing back into default DATA format
            data = data + net[l]['biases'].reshape(1,-1,1,1)                                    # 19: adding bias terms via broadcasting

        elif net[l]['type'] == 'relu': data = np.maximum(data, 0)                               # 20: RELU

        elif net[l]['type'] == 'pool':                                                          # 21: POOLING (MAX)
            data = pad(data, [0,0,0,0]+net[l]['pad'])                                           # 22: padding DATA in spatial dimensions (H and W) if needed
            data = unroll(data, (1,1,)+tuple(net[l]['pool'])).squeeze((4,5))                    # 23: rolling window view of DATA (N*C*H*W -> N*C*(H-h+1)*(W-w+1)*h*w)
            data = data[:,:,::net[l]['stride'][0],::net[l]['stride'][1]]                        # 24: striding (before computation)
            data = np.amax(data.reshape(data.shape[0:4]+(-1,)), -1)                             # 25: max pooling

        elif net[l]['type'] == 'normalize':                                                     # 26: LRN
            ndat = pad(np.square(data), [0,0]+2*[(net[l]['param'][0]-1)/2]+[0,0,0,0])           # 27: padding squared DATA in spectral dimension (C) to simplify boundary conditions
            ndat = unroll(ndat, (1,net[l]['param'][0],1,1)).squeeze((4,6,7))                    # 28: rolling window view of DATA (N*(C+2c)*H*W -> N*C*H*W*(2c+1))
            ndat = (net[l]['param'][1] + ndat.sum(-1)*net[l]['param'][2]) ** net[l]['param'][3] # 29: denominator
            data = data / ndat                                                                  # 30: element-wise normalization

        elif net[l]['type'] == 'softmax':                                                       # 31: SOFTMAX
            data = np.exp(data - np.amax(data, axis=1)[:,None,:,:])                             # 32: numerically-safe exponential
            data = data / data.sum(1)[:,None,:,:]                                               # 33: normalizing into probabilities

        else: raise StandardError('Operation in layer {0} undefined!'.format(str(l)))           # 34: throwing error for undefined layer types

    return data                                                                                 # 35: done
