# BEV Semantic Map
```
cd ~/git/bevnet/bevnet/network/ops
python3 setup.py develop

cd ~/git/bevnet
pip3 install -e ./

```
## Additional dependencies
```
pip install pytictac
pip install efficientnet-pytorch
```

run:
```
python bevnet/bevnet/network/bev_net.py
```


## Jonas Setup with venv
Clone repo and create venv:

```
cd ~/git; git clone git@github.com:leggedrobotics/bevnet.git -b dev/geometry_only
sudo apt install python3.8-venv
python3 -m venv ~/venv/bevnet
```

Add alias to .zshrc
```
alias venv_bevnet="source ~/venv/bevnet/bin/activate"
```

WITH CUDA 12.1 System installation -> The only installation we have to do before pip install the package because of the setup.py requiring torch.

```
pip3 install torch torchvision torchaudio # This should get the 12.1 CUDA version of TORCH 2.1.2
```

```
cd ~/git/bevnet
pip3 install -e ./
cd ~/git/bevnet/bevnet/network/ops
python3 setup.py develop
```

## Comments can be ignored
Packages not yet verified: 
```
tqdm
simple-parsing
scipy
scikit-learn
ros_numpy==0.0.5
Pillow==9.4.0
nuscenes-devkit==1.1.10
matplotlib
efficientnet-pytorch==0.7.0
```