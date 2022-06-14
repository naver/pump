# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

CODE_ROOT=`pwd`
if [ ! -e datasets ]; then
    echo "Error: missing datasets/ folder"
    echo "First, create a folder that can host (at least) 15 GB of data."
    echo "Then, create a soft-link named 'data' that points to it."
    exit -1
fi

# download some web images from the revisitop1m dataset
WEB_ROOT=datasets/revisitop1m
mkdir -p $WEB_ROOT
cd $WEB_ROOT
if [ ! -e 0d3 ]; then
    for i in {1..5}; do
        echo "Installing the web images dataset ($i/5)..."
        if [ ! -f revisitop1m.$i.tar.gz ]; then
            wget http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.$i.tar.gz
        fi
        tar -xzvf revisitop1m.$i.tar.gz
        rm -f revisitop1m.$i.tar.gz
    done
fi
cd $CODE_ROOT

# download SfM120k pairs
SFM_ROOT=datasets/sfm120k
mkdir -p $SFM_ROOT
cd $SFM_ROOT
if [ ! -e "ims" ]; then
    echo "Downloading the SfM120k dataset..."
    fname=ims.tar.gz
    if [ ! -f $fname ]; then
        wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz
    fi
    tar -xzvf $fname -C ims
    rm -f $fname
fi
if [ ! -e "corres" ]; then
    echo "Installing the SfM120k dataset..."
    fname=corres.tar.gz
    if [ ! -f $meta ]; then
        wget https://download.europe.naverlabs.com/corres.tar.gz
    fi
    tar -xzvf $fname
    rm -f $fname
fi
cd $CODE_ROOT

echo "Done!"
