#! /bin/bash
# brief: Import various CNN models from the web
# author: Andrea Vedaldi

# --------------------------------------------------------------------
# Caffe Reference Models
# --------------------------------------------------------------------

CAFFE_URL=http://dl.caffe.berkeleyvision.org/
CAFFE_GIT=https://github.com/BVLC/caffe/raw

wget -c -nc $CAFFE_URL/caffe_reference_imagenet_model
wget -c -nc $CAFFE_URL/caffe_ilsvrc12.tar.gz
wget -c -nc $CAFFE_GIT/8198585b4a670ee2d261d436ebecbb63688da617/examples/imagenet/imagenet_deploy.prototxt
tar xzvf caffe_ilsvrc12.tar.gz

out=refnet.pkl

python import-caffe.py \
	--caffe-variant=caffe \
	--preproc=caffe \
	--average-image="imagenet_mean.binaryproto" \
	--synsets="synset_words.txt" \
	"imagenet_deploy.prototxt" \
	"caffe_reference_imagenet_model" \
	"$out"
