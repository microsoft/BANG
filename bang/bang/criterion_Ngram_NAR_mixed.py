# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
import random


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion('ngram_language_loss_NAR_mixed')
class NgramLmMixedLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.nar_ratio = args.nar_ratio

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        parser.add_argument('--nar-ratio', default=0., type=float, metavar='D',
                            help='0: AR, 1: NAR, ')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # AR or NAR
        if random.random() > self.nar_ratio:
            flag_AR = True
        else:
            flag_AR = False

        if flag_AR:
            logits_list = model(**sample['net_input'], return_all_hiddens=False, flag_AR=True)[0]
            targets = model.get_targets(sample, [logits_list[0]])

            ngram = len(logits_list)
            # [B, ngram, T]
            expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
            for i in range(ngram):
                if i > 0 and self.disable_ngram_loss:
                    break

                padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
                if 'target_idx' in sample:
                    expend_targets[i, :, :] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
                else:
                    expend_targets[i, :, :] = targets
            targets = expend_targets

            logits = torch.cat(logits_list, dim=0)  # .view(ngram, *logits_list[0].size())

            lprobs = F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            )

            loss = F.nll_loss(
                lprobs,
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

            if self.eps > 0.:
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
                non_pad_mask = targets.ne(self.padding_idx).view(-1)
                smooth_loss = smooth_loss[non_pad_mask]
                smooth_loss = smooth_loss.sum()

                eps_i = self.eps / lprobs.size(-1)
                loss = (1. - self.eps) * loss + eps_i * smooth_loss

            sample_size = targets.ne(self.padding_idx).int().sum().item()

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'flag_AR': 1.0,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output
        else:
            net_output = model(**sample['net_input'], flag_AR=False)
            loss, nll_loss = self.compute_loss_label_smoothed_cross_entropy(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'flag_AR': 0.0,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output


    def compute_loss_label_smoothed_cross_entropy(self, model, net_output, sample, reduce=True):
        #print(net_output)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
