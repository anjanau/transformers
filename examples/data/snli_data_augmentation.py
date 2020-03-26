import pandas as pd

#code to get samples from snli train in the format needed by movie nli data

TRAIN_DATA_PATH = 'snli_1.0_train.txt'
df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')

entail = df[(df.gold_label == 'entailment')].reset_index()
contradict = df[(df.gold_label == 'contradiction')].reset_index()
contradict.gold_label = ['contradictory']*contradict.shape[0]

num_entail_samples, num_contradict_samples = 2000, 2000
e = entail.sample(n=num_entail_samples, random_state=1 )
c = contradict.sample(n=num_contradict_samples, random_state=1)

columns = ['sentence1', 'sentence2', 'gold_label']
c.to_csv('contradiction.csv', columns=columns, sep='\t', index=False, index_label=False, header=False)
e.to_csv('entailment.csv', columns=columns, sep='\t', index=False, index_label=False, header=False)