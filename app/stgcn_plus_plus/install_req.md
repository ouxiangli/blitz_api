conda create -n openmmlab python=3.7 -y
conda activate openmmlab
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
pip install mmdet==2.23.0 mmpose==0.28.0 mmcv==1.3.18

git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
pip install -r requirements.txt
# error -> pip install pillow==8.3.2
pip install -e .

pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html


pip install moviepy --upgrade
conda install scikit-learn

apt-get install libgl1-mesa-glx




<!-- new -->
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
conda install scikit-learn -y
pip install tensorflow-cpu==2.4
pip install mediapipe
pip install opencv-python

conda install -n openmmlab ipykernel --update-deps --force-reinstall

pip install fastapi
pip install "uvicorn[standard]"

apt-get install libgl1-mesa-glx