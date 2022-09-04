import glob

def formatter(s):
    return f'/JPEGImages/{s}.jpg /SegmentationClassAug/{s}.png'

# Run in data/split/
if __name__ == '__main__':
    voc_list = [l for l in glob.glob('./*voc*.txt')]

    for l in voc_list:
        with open(l, 'r') as f:
            data_list = f.read().strip().split('\n')
        
        if ' ' not in data_list[0]:
            with open(l, 'w') as f:
                data = [formatter(d) for d in data_list]
                f.write('\n'.join(data))