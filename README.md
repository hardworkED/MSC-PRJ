# Multitask Learning Approach in Continuous Human Affects and rPPG Estimation

> **Jun Huang Goh**<br>
> 220399308<br>
> MSc in Artificial Intelligence<br>
> Supervisor: Niki Maria Foteinopoulou<br>

## Installation

Install required packages:

```
pip install -r requirements.txt
```

## Model running
This work uses 80/20 split training:
```
python main8020.py
```

## AMIGOS dataset pre-processing
- **data/AMIGOS.py**
data preprocessing functions
- **data/run_\*.py**
data preprocessing running examples
- **data/datasets.py**
torch.utils.data.Dataset constructor for AMIGOS