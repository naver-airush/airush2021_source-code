import math
import time


class Meter(object):
    def __init__(self):
        self.init()

    def init(self):
        self.start = time.time()
        self.cnt_add = 0
        self.tot_loss = 0.
        self.cnt_sent = 0
        self.cnt_token = 0

    def add(self, loss, n_sent, n_token):
        self.cnt_add += 1
        self.tot_loss += loss * n_sent
        self.cnt_sent += n_sent
        self.cnt_token += n_token

    def average(self):
        loss_sent = self.tot_loss / self.cnt_sent if self.cnt_sent != 0 else 0.
        loss_token = self.tot_loss / self.cnt_token if self.cnt_token != 0 else 0.
        return loss_sent, loss_token

    def elapsed_time(self):
        return time.time() - self.start

    def print_str(self, time_avg=False):
        loss_sent, loss_token = self.average()
        et  = self.elapsed_time()
        time_str = f"{et * 1000. / self.cnt_add:6.2f} ms/batch" if time_avg else f"{et:6.2f} s"
        return f"{time_str} | loss_sent {loss_sent:6.2f} | token_ppl {math.exp(loss_token):6.2f}"


