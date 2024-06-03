import torch

def read_mse(path):
    # read
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    # parse: format: Batch: idx, Loss: mse 
    mse = []
    for line in lines:
        mse.append(float(line.split(' ')[-1]))
    return mse


if __name__ == "__main__":
    paths = [
        "/home/shx/zcy/proj/drbot/outputs/exp2/step10000/test_metrics.txt", 
        "/home/shx/zcy/proj/drbot/outputs/exp1/step10000/test_metrics.txt"        
    ]
    
    for path in paths:
        mse = read_mse(path)
        mse = torch.tensor(mse)
        print(mse.mean())

