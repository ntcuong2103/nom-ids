

import re
ids_dict = {line.strip().split('\t')[1]:re.sub(r'\[.*\]', '', line.strip().split('\t')[2]) for line in open('ids.txt', 'r').readlines() if not line.startswith('#') and len(line.strip().split('\t')) > 1}

### Recursive IDS


# vocab_ids_dict
def get_full_ids(c):
    seq = ids_dict.get(c, c)
    if len(seq) > 1 and c not in seq:
        return ''.join([get_full_ids(cc) for cc in seq])
    return c     


ids_exp_dict = {k: get_full_ids(k) for k in ids_dict}


# write to file
with open('ids_exp.txt', 'w') as f:
    f.writelines([f'{k}\t{v}\n' for k, v in ids_exp_dict.items()])


# build vocab
vocab_ids = set()
for k, v in ids_exp_dict.items():
    vocab_ids.update(v)


# write vocab ids full to file
with open('vocab_ids.txt', 'w') as f:
    f.write('\n'.join(sorted(vocab_ids)))