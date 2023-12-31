from visdom import Visdom
import numpy as np


class Visualizeer:
    def __init__(self, env='default', **kwargs):
        self.vis = Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        ''' self.plot('loss', 1.00) '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=str(name), update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def plot_line(self, name, x, y, **kwargs):
        self.vis.line(X=np.array([x]), Y=np.array([y]), win=str(name), update=None if x == 0 else 'append', **kwargs)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def img(self, name, img_, **kwargs):
        self.vis.images(img_,  # _.cpu().numpy(),
                        win=str(name),
                        opts=dict(title=name),
                        **kwargs)
