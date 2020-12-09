# import tensorflow as tf
from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from utils import doc_utils


class Trainer(object):
    def __init__(self, model_wrapper, data, config):
        self.is_QM9 = config.dataset_name == 'QM9'
        self.is_SimGNN = config.dataset_name in ['AIDS700nef', 'LINUX']
        self.best_val_loss = np.inf
        self.best_epoch = -1
        self.cur_epoch = 0

        self.model_wrapper = model_wrapper
        self.config = config
        self.data_loader = data

        self.optimizer = None
        if self.config.hyperparams.optimizer == 'momentum':
            self.optimizer = torch.optim.SGD(self.model_wrapper.model.parameters(),
                                             lr=self.config.hyperparams.learning_rate,
                                             momentum=self.config.hyperparams.momentum)
        elif self.config.hyperparams.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model_wrapper.model.parameters(),
                                              lr=self.config.hyperparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=config.hyperparams.decay_rate)

    def train(self):
        """
        Trains for the num of epochs in the config.
        :return:
        """
        for cur_epoch in range(self.cur_epoch, self.config.num_epochs, 1):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            self.cur_epoch = cur_epoch
            # validation step
            if self.config.val_exist:
                val_acc, val_loss = self.validate(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, val_acc, val_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            if not self.is_QM9:
                doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param num_epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize('train')
        self.model_wrapper.train()

        # initialize tqdm
        total_loss = 0.
        total_correct_labels_or_distances = 0.
        total_size = 0

        # Iterate over batches
        for cur_it in range(self.data_loader.num_iterations_train):
            # One Train step on the current batch
            if self.is_SimGNN:
                loss, correct_labels_or_distances, batch_size = self.train_step()
                total_size += batch_size
            else:
                loss, correct_labels_or_distances = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct_labels_or_distances += correct_labels_or_distances

        self.scheduler.step()

        if self.data_loader.train_size == -1: #dummy value for SimGNN
            self.data_loader.train_size = total_size

        loss_per_epoch = total_loss/self.data_loader.train_size
        if not self.is_QM9 and not self.is_SimGNN:
            acc_per_epoch = total_correct_labels_or_distances/self.data_loader.train_size
            print("\t\tEpoch-{}  loss:{:.4f} -- acc:{:.4f}\n".format(num_epoch, loss_per_epoch, acc_per_epoch))
            return acc_per_epoch, loss_per_epoch
        elif self.is_SimGNN:
            dist_per_epoch = (total_correct_labels_or_distances * self.data_loader.labels_std)/self.data_loader.train_size
            print(f"Epoch {num_epoch} train loss: {loss_per_epoch*1000:.3f} e-3 mean_distances: {dist_per_epoch*1000:.3f} e-3")
            return dist_per_epoch, loss_per_epoch
        else:
            dist_per_epoch = (total_correct_labels_or_distances * self.data_loader.labels_std)/self.data_loader.train_size
            print(f"Epoch-{num_epoch} loss:{loss_per_epoch:.4f} -- mean_distances: {dist_per_epoch}")
            return dist_per_epoch, loss_per_epoch

    def train_step(self):
        """

        :return: tuple of (loss, num_correct_labels or distances_array)
        """
        graphs, labels = self.data_loader.next_batch()
        loss, correct_labels_or_distances = self.model_wrapper.run_model_get_loss_and_results(graphs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_SimGNN:
            batch_size = graphs.shape[0]*(graphs.shape[0]-1);
            return loss.cpu().item(), correct_labels_or_distances, batch_size
        else:
            return loss.cpu().item(), correct_labels_or_distances

    def validate(self, epoch):
        """
        Perform forward pass on the model with the validation set
        :param epoch: Epoch number
        :return: (val_acc, val_loss) for benchmark graphs, (val_dists, val_loss) for QM9
        """
        # initialize dataset
        self.data_loader.initialize('val')
        self.model_wrapper.eval()

        # initialize tqdm
        # tt = tqdm(range(self.data_loader.num_iterations_val), total=self.data_loader.num_iterations_val,
        #           desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct_or_dist = 0.
        total_size = 0

        # Iterate over batches
        for cur_it in range(self.data_loader.num_iterations_val):
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            # label = np.expand_dims(label, 0)
            batch_size = graph.shape[0] * (graph.shape[0]-1)
            total_size += batch_size
            loss, correct_or_dist = self.model_wrapper.run_model_get_loss_and_results(graph, label)

            # update metrics returned from train_step func
            total_loss += loss.cpu().item()
            total_correct_or_dist += correct_or_dist

        if self.data_loader.val_size == -1:
            self.data_loader.val_size = total_size

        val_loss = total_loss/self.data_loader.val_size
        if self.is_QM9 or self.is_SimGNN:
            val_dists = (total_correct_or_dist*self.data_loader.labels_std)/self.data_loader.val_size
            if self.is_QM9:
                print(f"Val-{epoch} loss:{val_loss:.4f} - mean_distances:{val_dists}")
            else:
                print(f"Epoch {epoch}  val  loss: {val_loss*1000:.3f} e-3 mean_distances: {val_dists * 1000:.3f} e-3")

            # save best model by validation loss to be used for test set
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("New best validation score achieved.")
                self.model_wrapper.save(best=True, epoch=self.cur_epoch, optimizer=self.optimizer)
                self.best_epoch = epoch
            return val_dists, val_loss
        else:
            val_acc = total_correct_or_dist/self.data_loader.val_size
            return val_acc, val_loss

    def test(self, load_best_model=False):
        """
        Perform forward pass on the model for the test set
        :param load_best_model: Boolean. True for loading the best model saved, based on validation loss
        :return: (test_dists, test_loss)
        """
        # load best saved model
        if load_best_model:
            _optimizer_state_dict, _epoch = self.model_wrapper.load(best=True)

        # initialize dataset
        self.data_loader.initialize('test')
        self.model_wrapper.eval()

        # initialize tqdm

        total_loss = 0.
        total_dists = 0.
        total_size = 0

        # Iterate over batches
        for cur_it in range(self.data_loader.num_iterations_test):
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            # label = np.expand_dims(label, 0)
            batch_size = graph.shape[0] * (graph.shape[0]-1)
            total_size += batch_size
            loss, dists = self.model_wrapper.run_model_get_loss_and_results(graph, label)
            # update metrics returned from train_step func
            total_loss += loss.cpu().item()
            total_dists += dists

        if self.data_loader.test_size == -1:
            self.data_loader.test_size = total_size

        test_loss = total_loss/self.data_loader.test_size
        test_dists = (total_dists*self.data_loader.labels_std) / self.data_loader.test_size
        if self.is_SimGNN:
            print(f"Epoch {self.best_epoch}  val  loss: {test_loss * 1000:.3f} e-3 mean_distances: {test_dists * 1000:.3f} e-3")
        else:
            print(f"Test-{self.best_epoch}  loss:{test_loss:.4f} - mean_distances: {test_dists}")

        return test_dists, test_loss
