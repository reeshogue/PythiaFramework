import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import Transformer
from dataset import Dataset
from utils import *
import atexit
import torchvision.io as tvio
import random

class Trainer:
    def __init__(self, size=(16, 16), eval=False, nheads=6, nlayers=8):
        self.size = size
        self.nheads = nheads
        self.nlayers = nlayers
        size = 16,16
        self.size = size
        self.eval = eval
        
        print("Loading Net...")
        self.transformer = Transformer(size=size[0]*size[1]*3, nheads=nheads, nlayers=nlayers).cuda()
        print("Net has", get_nparams(self.transformer), "parameters.")
        # self.start_word = (tvio.read_image('../data/new.png', mode=tvio.image.ImageReadMode.RGB).float() / 255).flatten(start_dim=0).unsqueeze(0).cuda()
    def train(self):
        print("Beginning Training...")
        for epoch in range(1):
            for i, data in enumerate(self.dataloader):
                x, y = data
                temp = random.uniform(0, 0.9)
                x = x.cuda()
                y = y.cuda()
                self.optim.zero_grad()
                y_false = self.transformer(x)
                loss = F.mse_loss(y_false, y)
                print("Loss: {}".format(loss.item()), end='\r', flush=True)
                loss.backward()
                self.optim.step()
    def load(self, path='../saves/checkpoint.pt'):
        checkpoint = torch.load(path, map_location='cpu')
        del self.transformer
        self.transformer = Transformer(nheads=self.nheads, nlayers=self.nlayers, size=self.size[0]*self.size[1]*3)
        self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda()
        del checkpoint['model']
        if not self.eval:
            del self.optim
            self.optim = torch.optim.Adam(self.transformer.parameters(), lr=1e-5, betas=(0.9999, 0.99999))
            self.optim.load_state_dict(checkpoint['optim'])
            del checkpoint['optim']

    def evaluate(self, start=None):
        with torch.no_grad(): 
            if start is None:
                start = torch.zeros((3,self.size[0]*self.size[1]*3)).cuda()
            else:
                start = start.cuda()
            # start = torch.cat([self.start_word, start], dim=0)
            _, display, surface = init_camera_and_window()
            self.transformer.eval()
            acc = 0
            while True:
                get_events()
                if acc % 1024 == 0 and acc != 0:
                    print("Updating the screen.")
                    show_tensor(F.fold(start[-1024:].unsqueeze(0).transpose(1, 2), (512, 512), self.size, stride=self.size), display, surface)
                start = start.unsqueeze(0)
                outp_transformer, _ = self.transformer(start, 1.0)
                start = torch.cat([start, outp_transformer[:, -1:]], 1).squeeze(0)
                start = start[-2048:] if len(start) > 2048 else start
                acc += 1

if __name__ == '__main__':
    trainer = Trainer(eval=True)
    trainer.load()
    trainer.evaluate()