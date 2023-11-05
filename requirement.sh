git config --global alias.st status    # git status -> git st
git config --global user.name "yoosk526"
git config --global user.email "yoosk526@naver.com"
git config --global init.defaultBranch main    # "master" -> "main"
git config --global credential.helper cache
git remote add origin https://github.com/yoosk526/Training-for-SR.git

pip install ujson    # Loading annotations faster
pip install torchmetrics
pip install albumentations
pip uninstall -y opencv-python-headless
pip install opencv-python==4.5.5.62
pip install opencv-python-headless==4.5.5.62
apt-get update
apt-get install -y libgl1-mesa-glx

pip uninstall -y torchvision
pip install torchvision==0.15.1