import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteBlock(nn.Module):
    """
    Squeeze and Excitation block to enhance representational power.
    """
    def __init__(self, filters, reduction=4):
        super(SqueezeExciteBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters, filters // reduction)
        self.fc2 = nn.Linear(filters // reduction, filters)

    def forward(self, x):
        b, c, h, w = x.size()
        w_vec = self.global_pool(x).view(b, c)
        w_vec = F.relu(self.fc1(w_vec))
        w_vec = torch.sigmoid(self.fc2(w_vec))
        w_vec = w_vec.view(b, c, 1, 1)
        return x * w_vec


class ResBlock(nn.Module):
    def __init__(self, filters, use_bias=False, use_se=False, reduction=4):
        super(ResBlock, self).__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(filters)

        if use_se:
            self.se_block = SqueezeExciteBlock(filters, reduction)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se_block(out)
        out = F.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, action_space: int, config: dict):
        super(ResNet, self).__init__()
        self.action_space = action_space

        self.qubits = config["game_settings"]["N"]
        network_settings = config["network_settings"]
        self.n_blocks = network_settings.get("n_blocks", 20)
        self.filters = network_settings.get("filters", 256)
        self.use_bias = network_settings.get("use_bias", False)
        self.use_se = network_settings.get("use_se", True)
        self.dropout_rate = network_settings.get("dropout_rate", 0.1)
        self.value_hidden_units = network_settings.get("value_hidden_units", 256)
        self.policy_hidden_units = network_settings.get("policy_hidden_units", 256)

        # Initial convolution + BN
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=3, padding=1, bias=self.use_bias)
        self.bn1 = nn.BatchNorm2d(self.filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(filters=self.filters, use_bias=self.use_bias, use_se=self.use_se)
            for _ in range(self.n_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(self.filters, 2, kernel_size=1, bias=self.use_bias)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(2 * self.qubits * self.qubits, self.policy_hidden_units)
        self.policy_fc2 = nn.Linear(self.policy_hidden_units, action_space)

        # Value head
        self.value_conv = nn.Conv2d(self.filters, 1, kernel_size=1, bias=self.use_bias)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.qubits * self.qubits, self.value_hidden_units)
        self.value_fc2 = nn.Linear(self.value_hidden_units, self.value_hidden_units // 2)
        self.value_fc3 = nn.Linear(self.value_hidden_units // 2, 1)

        # Dropout layers
        self.policy_dropout1 = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.policy_dropout2 = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.value_dropout1 = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.value_dropout2 = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = torch.flatten(p, start_dim=1)
        if self.policy_dropout1 is not None:
            p = self.policy_dropout1(p)
        p = F.relu(self.policy_fc1(p))
        if self.policy_dropout2 is not None:
            p = self.policy_dropout2(p)
        p = self.policy_fc2(p)
        policy_output = F.softmax(p, dim=-1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = torch.flatten(v, start_dim=1)
        if self.value_dropout1 is not None:
            v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))
        if self.value_dropout2 is not None:
            v = self.value_dropout2(v)
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))

        return policy_output, v

    def predict(self, mat):
        # Convert input state to PyTorch tensor
        state = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            policy, value = self(state)
        return policy, value