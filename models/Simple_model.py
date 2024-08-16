import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRFIDModel(nn.Module):
    def __init__(self, num_epc, out_channels):
        super(SimpleRFIDModel, self).__init__()

        # MLP to process EPC (categorical data)
        self.epc_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # MLP to process RSSI and Phase data
        self.rssi_phase_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Final MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),  # Combining features from EPC and RSSI/Phase MLPs
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)  # Final output layer
        )

    def forward(self, epc, rssi_phase):
        # Process EPC input
        epc_features = self.epc_mlp(epc)

        # Process RSSI and Phase data
        rssi_phase_features = self.rssi_phase_mlp(rssi_phase)

        # Concatenate features from EPC and RSSI/Phase
        combined_features = torch.cat((epc_features, rssi_phase_features), dim=1)

        # Pass through final classifier
        output = self.classifier(combined_features)

        # Return log-softmax for classification
        return F.log_softmax(output, dim=1)