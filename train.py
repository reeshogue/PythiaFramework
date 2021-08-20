import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import Transformer
from dataset import Dataset
from utils import *
import atexit
import torchvision.io as tvio
import random
from optim import MADGRAD

class Trainer:
    def __init__(self, dry_run=True, eval=False, nheads=4, nlayers=4, optim_steps=1, lr=4e-3, patch_size=(16,16)):
        self.size = patch_size
        self.nheads = nheads
        self.nlayers = nlayers
        self.eval = eval
        self.dry_run = dry_run
        self.optim_steps = optim_steps
        
        print("Loading Net...")
        self.transformer = Transformer(size=self.size[0]*self.size[1]*3, nheads=nheads, nlayers=nlayers).cuda()
        print("Net has", get_nparams(self.transformer), "parameters.")
        self.dataset = Dataset(max_ctx_length=4096, size=self.size, dry_run=True)
        self.dataloader = D.DataLoader(self.dataset, shuffle=True, batch_size=1)
        self.optim =torch.optim.Adam(self.transformer.parameters(), lr=lr, betas=(0.9999, 0.99999))
    def train(self):
        print("Beginning Training...")
        for epoch in range(1):
            for i, data in enumerate(self.dataloader):
                x, y = data
                x = x.cuda()
                y = y.cuda()
                self.optim.zero_grad()
                y_false, sampling_loss = self.transformer(x)
                loss = F.binary_cross_entropy(y_false, y) + 0.4 * sampling_loss
                if i % 20 == 0:
                    print("Loss: {0}, Iteration: {1} its.".format(loss.item(), i), flush=True)
                print(loss.item())
                loss.backward()

                for _ in range(self.optim_steps):
                    self.optim.step()

    def save(self, path='../saves/checkpoint.pt'):
        torch.save({'optim':self.optim.state_dict(), 'model':self.transformer.state_dict()}, path)
    def load(self, path='../saves/checkpoint.pt'):
        checkpoint = torch.load(path, map_location='cpu')
        del self.transformer
        self.transformer = Transformer(nheads=self.nheads, nlayers=self.nlayers, size=self.size[0]*self.size[1]*3)
        self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda()
        del checkpoint['model']
        if not self.eval:
            del self.optim
            self.optim = torch.optim.Adam(self.transformer.parameters(), lr=4e-3, betas=(0.9999, 0.99999))
            self.optim.load_state_dict(checkpoint['optim'])
            del checkpoint['optim']

if __name__ == '__main__':
    trainer = Trainer(eval=False)
    atexit.register(lambda:trainer.save())
    trainer.train()