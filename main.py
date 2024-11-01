import glob
import os
import pickle
import sys
import time
import csv
from tqdm import tqdm
from util.arguments import get_parser
import yaml
import random
import numpy as np
import paddle
from tensorboardX import SummaryWriter
from util.util import import_class
from util.util import print_color
import paddle.optimizer as optimizer
from sklearn.metrics import confusion_matrix
from feeders.tools import random_rot
from util.loss import FocalLoss


def init_seed(seed):
    # paddle 如何设置random seed
    np.random.seed(seed)
    random.seed(seed)


loss_alpha = [0.015, 0.015, 0.015, 0.102, 0.015, 0.015, 0.04, 0.032, 0.015, 0.015, 0.033, 0.015, 0.102, 0.015, 0.098,
              0.038, 0.022, 0.05, 0.015, 0.025, 0.052, 0.019, 0.015, 0.015, 0.05, 0.072, 0.015, 0.015, 0.035, 0.015]


class Processor:
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.loss = None
        self.optimizer = None
        self.data_loader = dict()
        self.cur_time = None

        print_color("\n>>> Current phase {} <<<\n".format(self.arg.phase), color="red")

        if arg.phase == 'train' or arg.phase == 'eval':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir)
                if os.path.isdir(arg.model_saved_name):
                    print_color('log_dir: {} already exist'.format(arg.model_saved_name))
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        os.system('rm -rf ' + arg.model_saved_name)
                        print_color('Dir removed: {}'.format(arg.model_saved_name), color='red')
                    else:
                        print_color('Dir not removed: {}'.format(arg.model_saved_name), color='red')

                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_timer = 0
        self.early_stop = False
        self.best_acc_epoch = 0

        if not arg.cpu:
            if type(self.arg.device) is list and len(self.arg.device) > 1:
                paddle.distributed.init_parallel_env()
                self.model = paddle.DataParallel(self.model)
        self.save_arg()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if self.arg.phase == 'train' or self.arg.phase == 'eval':
            self.arg.train_feeder_args["bone"] = self.arg.bone
            self.arg.train_feeder_args["motion"] = self.arg.motion
            self.data_loader['train'] = paddle.io.DataLoader(
                dataset=Feeder(phase=self.arg.phase, **self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        if self.arg.phase == 'eval':
            self.arg.test_feeder_args["data_path"] = self.arg.train_feeder_args["data_path"]
            self.arg.test_feeder_args["label_path"] = self.arg.train_feeder_args["label_path"]

        self.arg.test_feeder_args["bone"] = self.arg.bone
        self.arg.test_feeder_args["motion"] = self.arg.motion
        self.data_loader['test'] = paddle.io.DataLoader(
            dataset=Feeder(phase=self.arg.phase, **self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)
        self.loss = paddle.nn.CrossEntropyLoss()
        # self.loss = FocalLoss(alpha=loss_alpha)

        if self.arg.phase == 'test' or self.arg.phase == 'predict':
            if self.arg.weights is None:
                self.print_log("the weights path must be specified when in phase {}".format(self.arg.phase))
                exit(-1)

        if self.arg.weights:
            # saved_filename format: runs-65-11830.pdparams
            try:
                self.global_step = int(self.arg.weights[:-9].split('-')[-1])
            except ValueError as e:
                print_color(e, color='red')
                self.global_step = 0

            self.print_log('Load weights from {}.'.format(self.arg.weights))
            state_dict = paddle.load(self.arg.weights)
            self.model.set_state_dict(state_dict)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optimizer.Momentum(
                parameters=self.model.parameters(),
                learning_rate=self.arg.base_lr,
                momentum=0.9,
                use_nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optimizer.Adam(
                parameters=self.model.parameters(),
                learning_rate=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            # write arg from command line and arg from arg_dict
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            self.optimizer.set_lr(lr)
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = "[ " + localtime + ' ] ' + string
        print(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.numpy())
            timer['model'] += self.split_time()

            predict_label = np.argmax(output.numpy(), axis=1)
            acc = paddle.mean((paddle.to_tensor(predict_label) == label).astype('float32'))
            acc_value.append(acc.numpy())
            self.train_writer.add_scalar('acc', acc.numpy(), self.global_step)
            self.train_writer.add_scalar('loss', loss.numpy(), self.global_step)

            # statistics
            self.lr = self.optimizer.get_lr()
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        self.print_log('\tMean training loss: {:.4f} Mean training acc: '
                       '{:.2f}%'.format(np.mean(loss_value), np.mean(acc_value) * 100))
        self.print_log('\tTime consumption: [Data]: {dataloader}, [Network]: {model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            paddle.save(state_dict,
                        self.arg.model_saved_name + '/runs-' + str(epoch + 1) + '-' + str(
                            int(self.global_step)) + '.pdparams')

    def eval(self, epoch, save_score=False, loader_name=('test',), wrong_file=None, result_file=None):
        f_w = None
        f_r = None
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label.numpy())
                with paddle.no_grad():
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.numpy())
                    loss_value.append(loss.numpy())

                    predict_label = np.argmax(output.numpy(), axis=1)
                    pred_list.append(predict_label)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label)
                    true = list(label.numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i].numpy()) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            # record best epoch
            if accuracy > self.best_acc:
                self.best_timer = 0
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
            else:
                if (epoch + 1) > self.arg.save_epoch:
                    # early stop
                    self.best_timer += 1
                    if self.best_timer > 15:
                        self.early_stop = True

            print_color('Accuracy: {} Models: {}.'.format(accuracy, self.arg.model_saved_name))
            if self.arg.phase == 'eval':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # 记录每个sample的score
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                # save score
                filename = '{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln)
                self.print_log("Saving score to file : {}".format(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(score_dict, f)

                # save acc for each class:
                label_list = np.concatenate(label_list)
                pred_list = np.concatenate(pred_list)
                confusion = confusion_matrix(label_list, pred_list)
                list_diag = np.diag(confusion)
                list_raw_sum = np.sum(confusion, axis=1)
                each_acc = list_diag / list_raw_sum

                filename = '{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln)
                self.print_log("Saving accuracy for each class to file : {}".format(filename))
                with open(filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["class", 'acc'])
                    writer.writerows(zip(range(len(each_acc)), each_acc))

    def predict(self, loader_name=('test',)):
        self.model.eval()
        all_predictions = []

        for ln in loader_name:
            pred_list = []
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                with paddle.no_grad():
                    output = self.model(data)
                    pred_list.append(output.numpy())

            all_predictions.extend(pred_list)

        all_predictions = np.concatenate(all_predictions, axis=0)

        filename = f'{self.arg.work_dir}/pred.npy'
        if os.path.exists(filename):
            self.print_log(f"Deleting existing prediction file: {filename}")
            os.remove(filename)
        self.print_log(f"Saving result of prediction probabilities to file: {filename}")
        np.save(filename, all_predictions)

    def start(self):
        if self.arg.phase == 'train' or self.arg.phase == 'eval':
            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])

            def count_parameters(model):
                size = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
                return size.item()

            self.print_log(f'# Parameters Size: {count_parameters(self.model)}')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.early_stop:
                    print_color("Early stop at {} epoch".format(epoch), color='red')
                    break

                save_model = ((epoch + 1) % self.arg.save_interval == 0 and (epoch + 1) > self.arg.save_epoch) \
                             or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)

                if self.arg.phase == 'eval':
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=('test',))

            if self.arg.phase == 'train':
                self.eval(epoch=0, save_score=True, loader_name=('test',))
            elif self.arg.phase == 'eval':
                # test the best model
                weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]

                state_dict = paddle.load(weights_path)
                self.model.set_state_dict(state_dict)

                wf = weights_path.replace('.pdparams', '_wrong.txt')
                rf = weights_path.replace('.pdparams', '_right.txt')
                self.arg.print_log = False
                self.eval(epoch=0, save_score=True, loader_name=('test',), wrong_file=wf, result_file=rf)
                self.arg.print_log = True

                self.print_log(f'Best accuracy: {self.best_acc}')
                self.print_log(f'Epoch number: {self.best_acc_epoch}')
                self.print_log(f'Model name: {self.arg.work_dir}')
                self.print_log(f'Model total number of params: {count_parameters(self.model)}')
                self.print_log(f'Weight decay: {self.arg.weight_decay}')
                self.print_log(f'Base LR: {self.arg.base_lr}')
                self.print_log(f'Batch Size: {self.arg.batch_size}')
                self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
                self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pdparams', '_wrong.txt')
            rf = self.arg.weights.replace('.pdparams', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.arg.save_score = True
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=('test',), wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

        elif self.arg.phase == 'predict':
            self.predict()


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.load(f,Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()
    init_seed(args.seed)
    processor = Processor(args)
    processor.start()