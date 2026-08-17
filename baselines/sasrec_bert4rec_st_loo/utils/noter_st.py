import os
from os.path import join
import time


class Noter(object):
    """ console printing and saving into files """
    def __init__(self, args):
        self.args = args

        self.t_start = time.time()
        self.f_log = join(args.path_log, f'{args.m}-st-{args.data}-{time.strftime("%m-%d-%H-%M-", time.localtime())}-'
                                         f'{str(args.device)[0] + str(args.device)[-1]}-{args.seed}-{str(args.l2)}.log')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        # welcome
        self.log_msg('\n' + '-' * 20 + f' Experiment: {self.args.m}-st ' + '-' * 20)
        self.log_settings()

    def write(self, msg):
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    def log_msg(self, msg):
        print(msg)
        self.write(msg)

    def log_settings(self):
        msg = (f'[Info] {self.args.m} (data:{self.args.data}, cuda:{self.args.cuda})\n'
               f'| eval_mode {self.args.eval_mode} |\n'
               f'| len_max {self.args.len_max} | n_attn {self.args.n_attn} | n_head {self.args.n_head} | dropout {self.args.dropout} |\n'
               f'| lr {self.args.lr:.2e} | l2 {self.args.l2:.2e} | lr_g {self.args.lr_g:.1f} | lr_p {self.args.lr_p} |\n'
               f'| seed {self.args.seed} |\n')
        self.log_msg(msg)

    def log_train(self, loss_enc, t_gap):
        msg = f'| tr  | los | {f"{loss_enc:.4f}"[:6]} | {t_gap:>5.1f}s |'
        self.log_msg(msg)

    def log_valid(self, res):
        msg = f'| val |  F  | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f} |'
        self.log_msg(msg)

    def log_test(self, res):
        msg = f'| te  |  F  | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f} |'
        self.log_msg(msg)

    def log_final_result(self, res):
        self.log_msg('\n' + '-' * 10 + f' Experiment ended ' + '-' * 10)
        self.log_settings()
        msg = (f'[ Info ] {self.args.m}-st ({(time.time() - self.t_start) / 60:.1f} min)\n'
               f'      |                A/B                |\n'
               f'      | hr5    | hr10   | ndcg5  | ndcg10 |  mrr   |\n'
               f'|  F  | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f} |\n')
        self.log_msg(msg)
