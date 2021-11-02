import pandas as pd
import os
import numpy as np
import hashlib


base_dir = 'C:/Users/gsmai/Documents/GitHub/yolov4/predicts640wHalfPenalty/cropped'
files = [base_dir + '/' + f for f in os.listdir(base_dir) if f.lower().endswith('.jpg')]

md5s = []

for f in files:
    print(f + ' '*100, end='\r')
    with open(f, 'rb') as fh:
        md5 = hashlib.md5()
        c = fh.read()
        md5.update(c)
        encoded = md5.hexdigest()
        md5s.append(encoded)

len(md5s)

df = pd.DataFrame({'file':files, 'md5':md5s})

is_dup = df.md5.duplicated()


df[is_dup]
df.file[df.md5 == '5069e4225a9355d5242baae829fde161'].values
df[is_dup].file[df[is_dup].md5 == '5069e4225a9355d5242baae829fde161'].values

for rid in range(len(df[is_dup])):
    row = df[is_dup].iloc[rid]
    dup_files = df.file[df.md5 == row.md5].values
    assert(len(dup_files)>1)
    print(dup_files)
    serial_nums = [int(f.split('_')[-2]) for f in dup_files]
    to_keep_id = np.argmin(serial_nums)
    for id_, f in enumerate(dup_files):
        if id_ != to_keep_id:
            print('Deleting %s' % f)
            if os.path.isfile(f):
                os.remove(f)


