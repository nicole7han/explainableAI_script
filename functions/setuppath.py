import glob
import pandas as pd
stim_path = "/Users/nicolehan/Documents/Research/Explainable AI/Stimuli"
blocks = glob.glob("{}/block*".format(stim_path))
blocks = [f for f in blocks if "xlsx" not in f]
blocks.sort()
stim_info = pd.read_excel('Stimuli/stim_info.xlsx')
for b in range(len(blocks)):
    df = pd.DataFrame(columns=['image', 'corrAns'])
    stims = glob.glob("{}/*.jpg".format(blocks[b]))

    for i in stims:
        df = df.append({'image':i, 'corrAns':1 if "target" in i else 0}, ignore_index=True)

    df.to_excel('Stimuli/block{}_stim_path.xlsx'.format(b+1), index=None)