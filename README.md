# Bayesian_Personalized_Ranking_Pytorch
Note that I use the movie lens datasets (ml-1m)
I used the code from https://github.com/guoyang9/BPR-pytorch

improment
  1. no negative sampe num in my case
  2. predict all negative samples for top k

""""
  class BPRData(data.Dataset):
    def __init__(self, users, items, candidates, num_items, is_training = True):
        super(BPRData, self).__init__()
        self.users = users
        self.items = items
        self.is_training = is_training
        self.cand = candidates
        self.num_items = num_items
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.is_training == True:
            while True:
                # Negative sample
                positive_list = self.cand[int(self.users[idx])]
                negative_sample = random.randint(0, self.num_items - 1)
                if negative_sample not in positive_list:
                    break

        user = self.users[idx]
        item_i = self.items[idx]
        item_j = negative_sample if \
                self.is_training else self.items[idx]

        return user, item_i, item_j
  
  """
