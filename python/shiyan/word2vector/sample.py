import numpy as np

import importlib,sys
importlib.reload(sys)

target = "../dataset/train_questions.txt"

rand_i = np.random.choice(range(36190),size=500,replace=False)
with open(target,encoding='UTF8') as f, open("../dataset/target.txt", "w",encoding='UTF8') as f2:
    count = 1
    for line in f:
        # print(f.read())
        if count in rand_i:
            f2.write(line)
        count += 1

