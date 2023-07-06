import json
from collections import defaultdict
import argparse
from pprint import pprint

mode = 'train'

with open('data.json') as infile:
    content = json.load(infile)

print(type(content))

outfile = open('train.txt', 'w')


messages = content['messages-so-far']

# pprint(messages)
options = content['options-for-correct-answers']
# print(len(options))
option = options[0]
correct_id = option['candidate-id']

print(option)
context = []

# pprint(messages[0])

for message in messages:
    text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
    context.append(text)

pprint(context)

outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), option['utterance'].strip()))



# write negative samples
# if args.mode != 'train':
#     negs = content['options-for-next']
#     for neg in negs:
#         if neg['candidate-id'] != correct_id:
#             outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg['utterance'].strip()))

if mode == 'train':
    negs = content['options-for-next']
    cnt = 0
    for neg in negs:
        if neg['candidate-id'] != correct_id:
            cnt += 1
            outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg['utterance'].strip()))
            if cnt == 15: # 16-1
                break