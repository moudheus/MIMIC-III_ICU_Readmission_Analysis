import os
#import shutil
import argparse
from sklearn.model_selection import KFold
import sys

header = 'stay,period_length,y_true'

print(sys.argv)
script, listfile_path, out_dir = sys.argv

os.makedirs(out_dir, exist_ok=True)  

patients = set()
with open(listfile_path, "r") as valset_file:
    for i, line in enumerate(valset_file):
        if i > 0:
            x = line.split(',')
            z = x[0].split('_')
            patients.add(z[0])
        
patients = list(patients)
with open(listfile_path, "r") as listfile:
    lines = listfile.readlines()[1:]


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
folds = []
for train_indices, test_indices in k_fold.split(patients):
    print('train_index:%s , test_index: %s ' % (train_indices, test_indices))
    folds.append(test_indices)
    
cvs=[[0,1,2,3,4,5,6,7,8,9],[2,3,0,1,4,5,6,7,8,9],[4,5,0,1,2,3,6,7,8,9],[6,7,0,1,2,3,4,5,8,9],[8,9,0,1,2,3,4,5,6,7]]


def write(index, fold, fold_lines):
    path = os.path.join(out_dir, f"{index}_{fold}_listfile801010.csv")
    with open(path, "w") as f:
        f.write(header)
        for line in fold_lines:
            f.write(line)

            
for id, cv in enumerate(cvs):
    train_lines = []
    for idx, f in enumerate(cv):
        if idx == 0:
            fold = folds[f]
            pa = [patients[x] for x in fold]
            test_lines = [x for x in lines if x[:x.find("_")] in pa]
        elif idx == 1:
            fold = folds[f]
            pa = [patients[x] for x in fold]
            val_lines = [x for x in lines if x[:x.find("_")] in pa]
        else:
            fold = folds[f]
            pa = [patients[x] for x in fold]
            train_lines += [x for x in lines if x[:x.find("_")] in pa]

    print(len(train_lines), len(val_lines), len(test_lines), len(lines))
    assert len(train_lines) + len(val_lines) + len(test_lines) == len(lines)

    write(id, 'train', train_lines)
    write(id, 'val', val_lines)
    write(id, 'test', test_lines)
