import torch

__all__ = ["LossManager", "LossManagerMulti"]


class LossManager:
    def __init__(self, cfg_layers):
        self.layer_dict = {}
        for l in cfg_layers.values():
            self.layer_dict[l.name] = l

    def compute(
        self, pred, target=None, aux=None, aux_config_dict=None, compute_loss=True
    ):
        """
        Input:
        pred: dict { "layer1": model_output , ... }
        target: stacked tensor of targets

        automaticially applies the loss scaling

        Returns:
            loss List[torch.Tensor([1.0]) , ...]
            updated torch.Tensor shape:=(BS,C,H,W) dtype=float
        """
        loss = []
        updated = []
        for j, (k, layer) in enumerate(self.layer_dict.items()):
            x = layer.pre_loss_function(pred[k])
            if compute_loss:
                loss.append(
                    layer.loss_function(x, target[:, j][:, None], aux, aux_config_dict)
                    * layer.loss_scale
                )
            x = layer.post_loss_function(x, aux)
            updated.append(x)

        return loss, torch.cat(updated, dim=1)


class LossManagerMulti:
    def __init__(self, cfg_layers):
        self.cfg_layers = cfg_layers
        self.layer_dict = {}

        for gridmap_key in cfg_layers.keys():
            self.layer_dict[gridmap_key] = {}
            for lname, layer in cfg_layers[gridmap_key].items():
                self.layer_dict[gridmap_key][lname] = layer

    def compute(
        self, pred, target=None, aux=None, aux_config_dict=None, compute_loss=True
    ):
        """
        Input:
        pred: dict {micro: dict {layer_namer: tensor, ...}, short: dict {layer_namer: tensor, ...} }
        target: dict of stacked tensor of targets
        aux: dict of stacked tensor of auxs
        aux_config_dict: dict of dict for aux indices
        automaticially applies the loss scaling

        Returns:
            loss -> dict of lists -> dict {micro: [], short: []}
            updated -> same as target format
        """
        loss = {}
        updated_preds = {}

        for gridmap_key in self.cfg_layers.keys():
            gridmap_loss = []
            updated = []
            prev_channel = 0
            for j, (k, layer) in enumerate(self.layer_dict[gridmap_key].items()):
                num_channel = layer.channels
                # print(f"gridmap_key is {gridmap_key} and layer is {layer}")
                x = layer.pre_loss_function(pred[gridmap_key][k])
                if compute_loss:
                    gridmap_loss.append(
                        layer.loss_function(
                            x,
                            target[gridmap_key][
                                :, prev_channel : prev_channel + num_channel
                            ],
                            aux[gridmap_key],
                            aux_config_dict[gridmap_key],
                        )
                        * layer.loss_scale
                    )
                x = layer.post_loss_function(
                    x, aux[gridmap_key], aux_config_dict[gridmap_key]
                )
                updated.append(x)
                # Update the channel counter
                prev_channel = prev_channel + num_channel

            updated_preds[gridmap_key] = torch.cat(updated, dim=1)
            loss[gridmap_key] = gridmap_loss

        return loss, updated_preds
