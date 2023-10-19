import torch


class AnomalyLoss(torch.nn.Module):
    def __init__(self):
        super(AnomalyLoss, self).__init__()

        # self._confidence_generator = ConfidenceGenerator(
        #     std_factor=confidence_std_factor,
        #     method=method,
        #     log_enabled=log_enabled,
        #     log_folder=log_folder,
        #     anomaly_detection=True,
        # )

    def forward(self, res: dict):
        losses = res["logprob"].sum(1) + res["log_det"]  # Sum over all channels, resulting in h*w output dimensions

        # if update_generator:
        #     confidence = self._confidence_generator.update(x=losses, x_positive=losses, step=step)

        # loss_aux["confidence"] = confidence

        loss_mean = -torch.mean(losses)
        loss_pred = losses

        return loss_mean, loss_pred
