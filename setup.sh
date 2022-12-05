sudo apt-get --assume-yes update
sudo apt-get --assume-yes install git man wget curl unzip zip fish htop tmux


echo "fish" >> .bashrc; fish

sudo apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*
    
sudo apt-get --assume-yes install ffmpeg libsm6 libxext6  -y

sudo apt-get update

##### If no conda
wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
bash Miniconda3-latest-Linux-x86_64.sh


conda install -c conda-forge fish
conda create --name siimcovid python=3.9.2
conda activate siimcovid

## Clone newest repo
git clone https://github.com/psui3905/COMS4995-Project.git