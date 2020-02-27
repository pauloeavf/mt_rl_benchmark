import numpy as np
import random
import os
import gc

def split_dataset(names, f_paths, valid_test_size, train_paths, valid_paths, test_paths, cap_train=-1):
    for i in range(2):
        print('Splitting %s corpus into train / validation / test sets...' % names[i])

        with open(f_paths[i], encoding='utf-8') as f:
            txt = f.readlines()

        if i == 0:
            idx_val_test = np.random.choice(len(txt), valid_test_size, replace=False)
            s_idx_val = set(idx_val_test[:valid_test_size//2])
            s_idx_test = set(idx_val_test[valid_test_size//2:])
            s_idx_val_test = set(idx_val_test)
        
        train = [sent for i, sent in enumerate(txt) if i not in s_idx_val_test]
        if cap_train > 0:
            random.shuffle(train)
            train = train[:cap_train]
            
        print('Saving %s trainig set...' % names[i])
        with open(train_paths[i], 'w', encoding='utf-8') as ftrain:
            ftrain.writelines('\n'.join(train))

        valid = [sent for i, sent in enumerate(txt) if i in s_idx_val]
        print('Saving %s valid set...' % names[i])
        with open(valid_paths[i], 'w', encoding='utf-8') as fvalid:
            fvalid.writelines('\n'.join(valid))

        test = [sent for i, sent in enumerate(txt) if i in s_idx_test]
        print('Saving %s test set...' % names[i])
        with open(test_paths[i], 'w', encoding='utf-8') as ftest:
            ftest.writelines('\n'.join(test))

        del txt, train, valid, test
        gc.collect()

if __name__ == '__main__':
	
	# OpenSubtitle
	os_dir = r'../data/os2018/'
	os_en = os.path.join(os_dir, 'OpenSubtitles.en-pt.en')
	os_pt = os.path.join(os_dir, 'OpenSubtitles.en-pt.pt')

	split_dataset(names=('OS EN', 'OS PT'), 
				  f_paths=(os_en, os_pt), 
				  valid_test_size=3000, 
				  train_paths=(os.path.join(os_dir, 'os_train.en'), os.path.join(os_dir, 'os_train.pt')), 
				  valid_paths=(os.path.join(os_dir, 'os_valid.en'), os.path.join(os_dir, 'os_valid.pt')), 
				  test_paths=(os.path.join(os_dir, 'os_test.en'), os.path.join(os_dir, 'os_test.pt')),
				  cap_train=150000)