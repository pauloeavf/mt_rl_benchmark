from __future__ import division
import lib

class Evaluator(object):
    def __init__(self, model, metrics, dicts, opt):
        self.model = model
        self.loss_func = metrics["nmt_loss"]
        self.sent_reward_func = metrics["sent_reward"]
        self.corpus_reward_func = metrics["corp_reward"]
        self.dicts = dicts
        self.max_length = opt.max_predict_length

    def eval(self, data, pred_file=None, src_file=None):
        self.model.eval()

        total_loss = 0
        total_words = 0
        total_sents = 0
        total_sent_reward = 0
        total_sari = 0

        all_preds = []
        all_targets = []
        all_src = []
        all_saris = []
        for i in range(len(data)):
            batch = data[i]
            targets = batch[1]

            attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            self.model.decoder.attn.applyMask(attention_mask)
            outputs = self.model(batch[:-1], True)


            weights = targets.ne(lib.Constants.PAD).float()
            num_words = weights.data.sum()
            _, loss = self.model.predict(outputs, targets, weights, self.loss_func)

            preds = self.model.translate(batch, self.max_length)
            preds = preds.t().tolist()
            src = batch[0][0].t().tolist()
            targets = targets.data.t().tolist()
            rewards, _ = self.sent_reward_func(preds, targets)

            # [self.dicts['tgt'].getLabel(w) for w in data.turk[55 * 8 + 7].tolist()]
            # [self.dicts["tgt"].getLabel(w) for w in batch[3][1][1].tolist()]
            # refs = [' '.join([self.dicts["tgt"].getLabel(w) for w in sent.tolist() if w not in [0, 3]]) for sent in batch[3][1]]
            # pred = " ".join([self.dicts["tgt"].getLabel(w) for w in preds[1] if not w in [0, 3]])
            # src_j = " ".join([self.dicts["tgt"].getLabel(w) for w in src[1] if not w in [0, 3]])

            # for i in range(batch[0][0].size()[1]):
            #     refs = [' '.join([self.dicts["tgt"].getLabel(w) for w in sent.tolist() if w not in [0, 3]]) for sent in batch[3][i]]
            #     pred = ' '.join([self.dicts["tgt"].getLabel(w) for w in preds[i] if not w in [0, 3]])
            #     src_j = ' '.join([self.dicts["src"].getLabel(w) for w in src[i] if not w in [0, 3]])
            #     all_saris.append(lib.SARIsent(src_j, pred, refs))

            all_preds.extend(preds)
            all_targets.extend(targets)
            all_src.extend(src)

            total_loss += loss
            total_words += num_words
            total_sent_reward += sum(rewards)
            # total_sent_reward += sum(all_saris)
            total_sents += batch[1].size(1)
            # total_sari += sum(all_saris)

        loss = total_loss / total_words
        sent_reward = total_sent_reward / total_sents
        corpus_reward = self.corpus_reward_func(all_preds, all_targets)

        if pred_file is not None:
            self._convert_and_report(data, pred_file, all_preds,
                (loss, sent_reward, corpus_reward), all_src, src_file)

        return loss, sent_reward, corpus_reward

    def _convert_and_report(self, data, pred_file, preds, metrics, src, src_file):
        preds = data.restore_pos(preds)
        with open(pred_file, "w") as f:
            for sent in preds:
                sent = lib.Reward.clean_up_sentence(sent, remove_unk=False, remove_eos=True)
                sent = [self.dicts["tgt"].getLabel(w) for w in sent]
                print(" ".join(sent), file=f)
        with open(src_file, 'w') as fsrc:
            for sent in src:
                sent = [self.dicts['src'].getLabel(w) for w in sent]
                print(" ".join(sent), file=fsrc)

        loss, sent_reward, corpus_reward = metrics
        print("")
        print("Loss: %.6f" % loss)
        print("Sentence reward: %.8f" % (sent_reward))
        print("Corpus reward: %.2f" % (corpus_reward * 100))
        print("Predictions saved to %s" % pred_file)


