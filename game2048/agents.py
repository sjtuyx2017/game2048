import numpy as np
import h5py
from keras.models import load_model

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.model = load_model('model_lkh.h5')
        #self.model1 = load_model('divided_model1.h5')
        #self.model2 = load_model('divided_model2.h5')
        #self.model3 = load_model('divided_model3.h5')
        #self.model4 = load_model('divided_model4.h5')
        #self.model5 = load_model('divided_model5.h5')

    def one_hot_encoding(self,arr):
        OUT_SHAPE = (4, 4)
        CAND = 16
        map_table = {2 ** i: i for i in range(1, CAND)}
        map_table[0] = 00
        result = np.zeros(shape=OUT_SHAPE + (CAND,), dtype=bool)
        for r in range(OUT_SHAPE[0]):
            for c in range(OUT_SHAPE[1]):
                result[r, c, map_table[arr[r, c]]] = 1
        return result

    def step(self):
        board = self.game.board

        enc = self.one_hot_encoding(board)

        probability = self.model.predict(np.array([enc]))[0]
        direction=np.argmax(probability)
        print("direction: ",direction)
        return direction
