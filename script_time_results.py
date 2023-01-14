from collections import defaultdict
import pickle

# Parameters
datasets = ['wikivitals', 'wikivitals-fr', 'wikischools']
betas = [8, 7, 6, 5]
ss = [8, 7, 6, 5]
meta_params = ['orderTrue', 'orderFalse', 'without u']  # In order: ordered_attributes, unordered_attributes, without U
res_dict = defaultdict(dict)

# Extract execution time from logs
for d in datasets:
    res_dict[d] = defaultdict(dict)
    for o in meta_params:
        res_dict[d][o] = defaultdict(list)
        for b in betas:
            for s in ss:
                if o in ['orderTrue', 'orderFalse']:
                    with open(f'output/log/max_support_inf/log_{d}_{b}_{s}_{o}') as f:
                        lines = f.readlines()
                        for l in lines:
                            if 'Total time:' in l:
                                time = float(l.split(' ')[2])
                                res_dict[d][o][b].append(float(time))
                else:
                    with open(f'output/log/without_u/max_support_inf/log_{d}_{b}_{s}_orderTrue') as f:
                        lines = f.readlines()
                        for l in lines:
                            if 'Total time:' in l:
                                time = float(l.split(' ')[2])
                                res_dict[d][o][b].append(float(time))

# Save result                            
with open('computation_times_dictionary_max_support_inf.pkl', 'wb') as f:
    pickle.dump(res_dict, f)

print(f'Dict ready!')