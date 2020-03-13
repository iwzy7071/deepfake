# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler
from bisect import bisect_right

@register_lr_scheduler('fixed')
class FixedSchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        warmup_end_lr = args.max_lr

        self.min_lr = args.warmup_init_lr
        self.max_lr = args.max_lr
        self.steps_per_epochs = args.steps_per_epochs
        assert self.max_lr > self.min_lr, 'max_lr must be more than lr'


        if args.warmup_updates > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        else:
            self.lr_step = 1

        self.warmup_updates = args.warmup_updates
        self.lr_shrink = args.lr_shrink

        # initial learning rate
        self.lr = args.warmup_init_lr

        self.lr_steps = [int(item) for item in args.lr_steps.split(',')]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')

        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--max-lr', default=0.4, type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--max-update', default=10000, type=int, metavar='LR',
                            help='update the learning rate linearly for the first N updates')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            # warmup 5 epochs
            cur_epochs = int(num_updates / self.steps_per_epochs)

            lr_shrink = self.lr_shrink ** bisect_right(self.lr_steps, cur_epochs)
            #lr_shrink = self.lr_shrink ** (cur_epochs // 30)
            self.lr = self.max_lr * lr_shrink

        # self.optimizer.set_lr(self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    # def step(self, epoch, val_loss=None):
    #     """Update the learning rate at the end of the given epoch."""
    #     super().step(epoch, val_loss)
    #     # we don't change the learning rate at epoch boundaries
    #     return self.optimizer.get_lr()
    #
    #
    # def step_update(self, num_updates):
    #     """Update the learning rate after each update."""
    #     if self.args.warmup_updates > 0 and num_updates <= self.args.warmup_updates:
    #         self.warmup_factor = num_updates / float(self.args.warmup_updates)
    #         self.optimizer.set_lr(self.warmup_factor * self.lr)
    #     return self.optimizer.get_lr()