import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from .feedforward_networks import MLP

from transformers import logging
logging.set_verbosity_error()

class SciBERT(nn.Module):

    def __init__(self,  num_labels: int, freeze_layer_count: int = 0,
                 freeze_embeddings: bool = False, freeze_pooler: bool = False,
                 classifier_layer_sizes: list = None, *args, **kwargs):

        super(SciBERT, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased',
                                                                            num_labels=num_labels, *args, **kwargs)

        if classifier_layer_sizes:
            self.model.classifier = MLP(in_features=768, hidden_layer_sizes=classifier_layer_sizes,
                                        num_classes=num_labels)

        frozen_modules = []
        if freeze_layer_count != 0:

            # -1 indicates we freeze all embeddings
            if freeze_layer_count == -1:
                freeze_layer_count = len(self.model.base_model.encoder.layer)

            # freezes the specified number of layers of the encoder
            frozen_modules += self.model.base_model.encoder.layer[:freeze_layer_count]

            # freezing layers will also freeze embeddings
            freeze_embeddings = True

        if freeze_embeddings:
            frozen_modules.append(self.model.base_model.embeddings)

        if freeze_pooler:
            frozen_modules.append(self.model.base_model.pooler)

        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(name)

        import pdb; pdb.set_trace()

    def forward(self, input, output_classifier_features=False):

        # moves all inputs to the device used by the model
        if input["input_ids"].device != self.model.device:
            for key, value in input.items():
                input[key] = input[key].to(self.model.device)

        if output_classifier_features == True:
            # returns classifier output and pooled last hidden state
            base_model_output = self.model.base_model(**input).pooler_output
            return self.model.classifier(base_model_output), base_model_output
        else:
            return self.model(**input).logits
