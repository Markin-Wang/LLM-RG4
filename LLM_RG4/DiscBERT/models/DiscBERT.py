import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
import torch.nn.functional as F
from models.bert_model import hall_labeler
from sklearn.metrics import precision_recall_fscore_support
import csv
import numpy as np

class DiscBERT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.labeler = hall_labeler(0.1,self.args)
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.predict_study_id_list = []
        self.predict_report_list = []
        self.predict_ps_list = []
        self.predict_pp_list = []
        self.predict_view_list = []
        self.predict_comm_list = []
        self.predictroad = args.predictroad

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def CE_loss(self, x, y):
        #
        y = y.view(-1).long()
        loss = nn.CrossEntropyLoss()
        return loss(x,y)

    def forward(self, samples):
        report = samples["report"]
        ps_label = samples["prior_study"]
        pp_label = samples["prior_proc"]
        view_label = samples["view"]
        comm_label = samples["comm"]
        ps_predict, pp_predict, view_predict, comm_predict = self.labeler(report)
        ps_loss = self.CE_loss(ps_predict,ps_label)
        pp_loss = self.CE_loss(pp_predict, pp_label)
        view_loss = self.CE_loss(view_predict,view_label)
        comm_loss = self.CE_loss(comm_predict,comm_label)
        loss = (ps_loss + pp_loss + view_loss + comm_loss)/4.0
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        state_dict = self.state_dict()
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'pths'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'pths',
            "checkpoint_epoch{}_step{}_f1{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        report = samples["report"]
        ps_label = samples["prior_study"]
        pp_label = samples["prior_proc"]
        view_label = samples["view"]
        comm_label = samples["comm"]
        ps_predict, pp_predict, view_predict, comm_predict = self.labeler(report)

        ps_pred_labels = torch.argmax(ps_predict, dim=1)
        pp_pred_labels = torch.argmax(pp_predict, dim=1)
        view_pred_labels = torch.argmax(view_predict, dim=1)
        comm_pred_labels = torch.argmax(comm_predict, dim=1)

        ps_labels = ps_label.view(-1).cpu()
        pp_labels = pp_label.view(-1).cpu()
        view_labels = view_label.view(-1).cpu()
        comm_labels = comm_label.view(-1).cpu()

        ps_preds = ps_pred_labels.cpu()
        pp_preds = pp_pred_labels.cpu()
        view_preds = view_pred_labels.cpu()
        comm_preds = comm_pred_labels.cpu()
        self.val_step_outputs.append((ps_labels, ps_preds, pp_labels, pp_preds, view_labels, view_preds, comm_labels, comm_preds))

    def on_validation_epoch_end(self):
        ps_labels_all, ps_preds_all = [], []
        pp_labels_all, pp_preds_all = [], []
        view_labels_all, view_preds_all = [], []
        comm_labels_all, comm_preds_all = [], []

        for outputs in self.val_step_outputs:
            ps_labels_all.append(outputs[0])
            ps_preds_all.append(outputs[1])
            pp_labels_all.append(outputs[2])
            pp_preds_all.append(outputs[3])
            view_labels_all.append(outputs[4])
            view_preds_all.append(outputs[5])
            comm_labels_all.append(outputs[6])
            comm_preds_all.append(outputs[7])

        ps_labels_all = torch.cat(ps_labels_all)
        ps_preds_all = torch.cat(ps_preds_all)
        pp_labels_all = torch.cat(pp_labels_all)
        pp_preds_all = torch.cat(pp_preds_all)
        view_labels_all = torch.cat(view_labels_all)
        view_preds_all = torch.cat(view_preds_all)
        comm_labels_all = torch.cat(comm_labels_all)
        comm_preds_all = torch.cat(comm_preds_all)

        ps_precision, ps_recall, ps_f1, _ = precision_recall_fscore_support(ps_labels_all, ps_preds_all, average='binary')
        pp_precision, pp_recall, pp_f1, _ = precision_recall_fscore_support(pp_labels_all, pp_preds_all, average='binary')
        view_precision, view_recall, view_f1, _ = precision_recall_fscore_support(view_labels_all, view_preds_all, average='binary')
        comm_precision, comm_recall, comm_f1, _ = precision_recall_fscore_support(comm_labels_all, comm_preds_all, average='binary')

        mean_f1 = (ps_f1 + pp_f1 + view_f1 + comm_f1) / 4.0

        print(f"Prior Study - Precision: {ps_precision:.4f}, Recall: {ps_recall:.4f}, F1: {ps_f1:.4f}")
        print(f"Prior Proc  - Precision: {pp_precision:.4f}, Recall: {pp_recall:.4f}, F1: {pp_f1:.4f}")
        print(f"View        - Precision: {view_precision:.4f}, Recall: {view_recall:.4f}, F1: {view_f1:.4f}")
        print(f"Comm        - Precision: {comm_precision:.4f}, Recall: {comm_recall:.4f}, F1: {comm_f1:.4f}")
        print(f"Mean F1: {mean_f1:.4f}")

        if self.trainer.local_rank == 0:
            if mean_f1 > self.val_score:
                self.save_checkpoint(mean_f1)
                self.val_score = mean_f1
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        report = samples["report"]
        ps_label = samples["prior_study"]
        pp_label = samples["prior_proc"]
        view_label = samples["view"]
        comm_label = samples["comm"]
        ps_predict, pp_predict, view_predict, comm_predict = self.labeler(report)

        ps_pred_labels = torch.argmax(ps_predict, dim=1)
        pp_pred_labels = torch.argmax(pp_predict, dim=1)
        view_pred_labels = torch.argmax(view_predict, dim=1)
        comm_pred_labels = torch.argmax(comm_predict, dim=1)

        ps_labels = ps_label.view(-1).cpu()
        pp_labels = pp_label.view(-1).cpu()
        view_labels = view_label.view(-1).cpu()
        comm_labels = comm_label.view(-1).cpu()

        ps_preds = ps_pred_labels.cpu()
        pp_preds = pp_pred_labels.cpu()
        view_preds = view_pred_labels.cpu()
        comm_preds = comm_pred_labels.cpu()
        self.test_step_outputs.append(
            (ps_labels, ps_preds, pp_labels, pp_preds, view_labels, view_preds, comm_labels, comm_preds))

    def on_test_epoch_end(self):
        ps_labels_all, ps_preds_all = [], []
        pp_labels_all, pp_preds_all = [], []
        view_labels_all, view_preds_all = [], []
        comm_labels_all, comm_preds_all = [], []

        for outputs in self.test_step_outputs:
            ps_labels_all.append(outputs[0])
            ps_preds_all.append(outputs[1])
            pp_labels_all.append(outputs[2])
            pp_preds_all.append(outputs[3])
            view_labels_all.append(outputs[4])
            view_preds_all.append(outputs[5])
            comm_labels_all.append(outputs[6])
            comm_preds_all.append(outputs[7])

        ps_labels_all = torch.cat(ps_labels_all)
        ps_preds_all = torch.cat(ps_preds_all)
        pp_labels_all = torch.cat(pp_labels_all)
        pp_preds_all = torch.cat(pp_preds_all)
        view_labels_all = torch.cat(view_labels_all)
        view_preds_all = torch.cat(view_preds_all)
        comm_labels_all = torch.cat(comm_labels_all)
        comm_preds_all = torch.cat(comm_preds_all)

        ps_precision, ps_recall, ps_f1, _ = precision_recall_fscore_support(ps_labels_all, ps_preds_all,
                                                                            average='binary')
        pp_precision, pp_recall, pp_f1, _ = precision_recall_fscore_support(pp_labels_all, pp_preds_all,
                                                                            average='binary')
        view_precision, view_recall, view_f1, _ = precision_recall_fscore_support(view_labels_all, view_preds_all,
                                                                                  average='binary')
        comm_precision, comm_recall, comm_f1, _ = precision_recall_fscore_support(comm_labels_all, comm_preds_all,
                                                                                  average='binary')

        mean_f1 = (ps_f1 + pp_f1 + view_f1 + comm_f1) / 4.0
        mean_precision = (ps_precision + pp_precision + view_precision + comm_precision) / 4.0
        mean_recall = (ps_f1 + pp_recall + view_recall + comm_recall) / 4.0

        print(f"Prior Study - Precision: {ps_precision:.4f}, Recall: {ps_recall:.4f}, F1: {ps_f1:.4f}")
        print(f"Prior Proc  - Precision: {pp_precision:.4f}, Recall: {pp_recall:.4f}, F1: {pp_f1:.4f}")
        print(f"View        - Precision: {view_precision:.4f}, Recall: {view_recall:.4f}, F1: {view_f1:.4f}")
        print(f"Comm        - Precision: {comm_precision:.4f}, Recall: {comm_recall:.4f}, F1: {comm_f1:.4f}")
        print(f"Mean F1: {mean_f1:.4f}")
        print(f"Mean precision: {mean_precision:.4f}")
        print(f"Mean recall: {mean_recall:.4f}")


    def predict_step(self, samples, batch_idx):
        study_id = samples["study_id"]
        report = samples["report"]
        ps_predict, pp_predict, view_predict, comm_predict = self.labeler(report)

        ps_pred_labels = torch.argmax(ps_predict, dim=1)
        pp_pred_labels = torch.argmax(pp_predict, dim=1)
        view_pred_labels = torch.argmax(view_predict, dim=1)
        comm_pred_labels = torch.argmax(comm_predict, dim=1)

        ps_preds = ps_pred_labels.cpu().tolist()
        pp_preds = pp_pred_labels.cpu().tolist()
        view_preds = view_pred_labels.cpu().tolist()
        comm_preds = comm_pred_labels.cpu().tolist()

        self.predict_study_id_list += study_id
        self.predict_report_list += report
        self.predict_ps_list += ps_preds
        self.predict_pp_list += pp_preds
        self.predict_view_list += view_preds
        self.predict_comm_list += comm_preds

    def on_predict_epoch_end(self):
        filename = self.predictroad.replace('.csv','_hall_result.csv')
        length = len(self.predict_study_id_list)
        print('predict_ps_list:' + str(np.sum(self.predict_ps_list)/length))
        print('predict_pp_list:' + str(np.sum(self.predict_pp_list)/length))
        print('predict_view_list:' + str(np.sum(self.predict_view_list)/length))
        print('predict_comm_list:' + str(np.sum(self.predict_comm_list)/length))

        predict_ps_list = np.array(self.predict_ps_list)
        predict_pp_list = np.array(self.predict_pp_list)
        predict_view_list = np.array(self.predict_view_list)
        predict_comm_list = np.array(self.predict_comm_list)

        matrix = np.vstack((predict_ps_list, predict_pp_list, predict_view_list, predict_comm_list)).T
        rows_with_1 = np.any(matrix == 1, axis=1)

        num_rows_with_1 = np.sum(rows_with_1)

        print('Number of rows with at least one 1:' + str(num_rows_with_1/length))

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入两列数据
            writer.writerow(
                ['predict_study_id_list', 'predict_report_list', 'predict_ps_list', 'predict_pp_list', 'predict_view_list',
                 'predict_comm_list'])
            writer.writerows(
                zip(self.predict_study_id_list, self.predict_report_list, self.predict_ps_list, self.predict_pp_list,
                    self.predict_view_list, self.predict_comm_list))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()