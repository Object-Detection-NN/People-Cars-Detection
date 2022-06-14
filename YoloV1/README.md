# People-Cars-Detection Yolo V1


# Getting started
1. Clone repo
```
git clone https://github.com/Object-Detection-NN/People-Cars-Detection.git
```

2. Setup environment
```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

3. Generate datasets using [load_dataset.py](load_dataset.py) you can set dataset's parameters in script
```
python load_dataset.py
```

4. Train model using [train.py](train.py) you can set training parameters in script
```
python train.py
```
