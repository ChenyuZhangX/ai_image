import torch
import matplotlib.pyplot as plt
import os

path = '/mnt/zhangchenyu/proj/drbot/outputs/exp7/step3000'

def parse(path, sets = 400):
    # read 
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    # Format example --- Scene: 1 Set: 100 MSE: 0.44787976145744324 VAR: 0.43572136759757996
    Scenes = []
    

    for idx in range(3):
        mses = []
        vars = []
        for line in lines[idx * sets : idx * sets + 400]:
            scene = int(line.split(' ')[1])
            mse = float(line.split(' ')[5])
            var = float(line.split(' ')[7])
            mses.append(mse)
            vars.append(var)
        mses = torch.tensor(mses)
        vars = torch.tensor(vars)
        Scenes.append({'scene': scene, 'mse': mses, 'var': vars})
    return Scenes

if __name__ == "__main__":
    Scenes = parse(os.path.join(path, 'test_metrics.txt'))

    # plot mse and var
    for idx in range(3):
        
        print('Scene 2{}'.format(Scenes[idx]['scene']))
        print('mse mean: ', Scenes[idx]['mse'].mean())
        print('var mean: ', Scenes[idx]['var'].mean())

        plt.figure()
        # set size
        plt.figure(figsize=(10, 6))

        plt.title('Scene 2{}'.format(Scenes[idx]['scene']))
        plt.plot(Scenes[idx]['mse'], label='mse')
        plt.plot(Scenes[idx]['var'], label='var')
        
        # mean and max
        plt.axhline(Scenes[idx]['mse'].mean(), color='r', linestyle='--', label=f'mean mse {Scenes[idx]["mse"].mean(): .2f}')
        plt.axhline(Scenes[idx]['var'].mean(), color='g', linestyle='--', label=f'mean var {Scenes[idx]["var"].mean(): .2f}')

        plt.axhline(Scenes[idx]['mse'].max(), color='r', linestyle='-.', label=f'max mse {Scenes[idx]["mse"].max(): .2f}')
        plt.axhline(Scenes[idx]['var'].max(), color='g', linestyle='-.', label=f'max var {Scenes[idx]["var"].max(): .2f}')
        

        plt.legend()
        plt.savefig(os.path.join(path, 'scene2{}.png'.format(idx + 1)))
        # close
        plt.close()