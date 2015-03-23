import glob
import pickle
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import convnet

# Specify Dataset

dataset = '/home/kenyon/Research/Datasets/ILSVRC2012_img_val/*.JPEG'
dataset = sorted(glob.glob(dataset))[0:500]

# Load Network

print 'Loading Network'
network = pickle.load(open('refnet.pkl','rb'))
results = []

imsize = np.array([227.0, 227.0])
border = np.array([ 29.0,  29.0])

# Process Dataset

itlist = range(0, len(dataset), 100)

for i in itlist:
    print 'Processing Stack ' + str(int(i/100)+1) + '/' + str(len(itlist))
    imstksz = len(dataset)-i if (i==itlist[-1]) else 100
    imstack = imstksz * [None]

    for j in xrange(imstksz):
        imstack[j] = imread(dataset[i+j], mode='RGB')[:,:,::-1]
        imstack[j] = imresize(imstack[j], np.round(max((imsize+border)/imstack[j].shape[0:2])*np.array(imstack[j].shape[0:2])).astype(int))

        croploc = np.floor((np.array(imstack[j].shape[0:2])-imsize) / 2)
        imstack[j] = imstack[j][croploc[0]:croploc[0]+imsize[0], croploc[1]:croploc[1]+imsize[1]]
        imstack[j] = imstack[j].transpose(2,0,1)[None,:,:,:]
        imstack[j] = imstack[j].astype(np.float32) - network['normalization'].astype(np.float32)

    imstack = np.concatenate(tuple(imstack), 0)
    ftstack = convnet.forward(imstack, network['layers']).squeeze()

    results.append(np.concatenate((np.argsort(ftstack)[:,-1:-6:-1], np.sort(ftstack)[:,-1:-6:-1]), 1))

results = np.concatenate(tuple(results), 0)

# Generate Results in HTML

outfile = open('results.html', 'w')
outline = '<td align="center"><img data-original="file://{0}" height="100" width="100" class="img-rounded"></td><td>{1}</td>\n'

outfile.write('<!DOCTYPE html>\n\
               <html lang="en">\n\
               <head>\n\
               <title>Results</title>\n\
               <meta charset="utf-8">\n\
               <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">\n\
               <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>\n\
               <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery.lazyload/1.9.1/jquery.lazyload.min.js"></script>\n\
               <script>$(function() {$("img.img-rounded").lazyload();});</script>\n\
               </head>\n\
               <body>\n\
               <div class="container-fluid">\n\
               <table class="table table-condensed table-bordered">\n')

for i in xrange(results.shape[0]):

    outentry = ''
    for j in xrange(5):
        outentry = outentry + str(int(results[i,j])).zfill(3)+': '
        outentry = outentry + network['classes'][results[i,j]] + ' '
        outentry = outentry + '(<i>P</i> = ' + '{:6f}'.format(results[i,j+5]) + ')<br>'

    if (i%5)==0: outfile.write('<tr>\n')
    outfile.write(outline.format(dataset[i], outentry))
    if (i%5)==4 or i==(results.shape[0]-1): outfile.write('</tr>\n')

outfile.write('</table> \n\
               </div> \n\
               </body> \n\
               </html>')
