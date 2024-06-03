import pickle


if __name__ == "__main__":
    path = '/mnt/zhangchenyu/proj/drbot/outputs/exp7/'
    with open(path + 'config.pkl', 'rb') as f:
        cfg = pickle.load(f)
    # format printing dict
    print(cfg)
    