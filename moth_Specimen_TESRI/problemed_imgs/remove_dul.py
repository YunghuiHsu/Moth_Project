from pathlib import Path
import os

dir_b = Path('Broken')
path_b = list(dir_b.glob('**/*'))
len(path_b)
name_b = set(path.stem for path in path_b)
len(name_b)


dir_f = Path('Notfull')
path_f = list(dir_f.glob('**/*'))
len(path_f)
name_f = set(path.stem for path in path_f)
len(name_f)

inter = name_b & name_f
len(inter)
inter

for idx, path in enumerate(path_f):
    name = path.stem
    if name in inter:
        print(idx, name)
        path.unlink()

# ======================================================================
list(dir_b.iterdir())

dir_20 = Path('Broken/Broken_10-20_')
dir_40 = Path('Broken/Broken__20-40_')
dir_over40 = Path('Broken/Broken_over40_')

path_20, path_40, path_over40 = list(dir_20.glob(
    '**/*')), list(dir_40.glob('**/*')), list(dir_over40.glob('**/*'))
name_20, name_40, name_over40 = set(path.stem for path in path_20), set(
    path.stem for path in path_40), set(path.stem for path in path_over40)
inter = name_40 & name_over40
len(inter)
inter
