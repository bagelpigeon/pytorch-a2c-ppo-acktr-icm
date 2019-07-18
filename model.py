import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import get_distribution
from utils import orthogonal

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_head(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.orthogonal(m.weight)
        m.weight.data.mul_(nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)

class NatureHead(torch.nn.Module):
    ''' DQN Nature 2015 paper
        input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
    '''
    def __init__(self, n):
        super(NatureHead, self).__init__()
        self.conv1 = nn.Conv2d(n, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.dense = nn.Linear(32 * 7 * 7, 512)
        self.output_size = 512
    
    def forward(self, state):
        output = F.relu(self.conv1(state))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.dense(output.view(-1, 32 * 7 * 7)))
        return output

class UniverseHead(torch.nn.Module):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    def __init__(self, n):
        super(UniverseHead, self).__init__()
        self.conv1 = nn.Conv2d(n, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.output_size = 288

    def forward(self, state):
        output = F.elu(self.conv1(state))
        output = F.elu(self.conv2(output))
        output = F.elu(self.conv3(output))
        output = F.elu(self.conv4(output))
        return output.view(-1, self.output_size)

class ICM(torch.nn.Module):
    def __init__(self, action_space, state_size, num_inputs=1, cnn_head=True):
        super(ICM, self).__init__()
        if cnn_head:
            self.head = NatureHead(num_inputs)
        if action_space.__class__.__name__ == "Discrete":
            action_space = action_space.n
        else:
            action_space = action_space.shape[0] * 2
        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_space, 256),
            nn.ReLU(),
            nn.Linear(256, state_size))
        self.inverse_model = nn.Sequential(
            nn.Linear(state_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.ReLU())

    def forward(self, state, next_state, action):
        if hasattr(self, 'head'):
            phi1 = self.head(state)
            phi2 = self.head(next_state)
        else:
            phi1 = state
            phi2 = next_state
        phi2_pred = self.forward_model(torch.cat([action, phi1], 1))
        action_pred = F.softmax(self.inverse_model(torch.cat([phi1, phi2], 1)), -1)
        return action_pred, phi2_pred, phi1, phi2

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """
        
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)
        
        action = self.dist.sample(hidden_actor, deterministic=deterministic)

        action_log_probs, dist_entropy, action_probs = self.dist.logprobs_and_entropy(hidden_actor, action)
        value = self.critic_linear(hidden_critic)
        
        return value, action, action_log_probs, states, action_probs

    def get_value(self, inputs, states, masks):        
        hidden_critic, _, states = self(inputs, states, masks)
        value = self.critic_linear(hidden_critic)
        return value
    
    def evaluate_actions(self, inputs, states, masks, actions):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action_log_probs, dist_entropy, action_probs = self.dist.logprobs_and_entropy(hidden_actor, actions)
        value = self.critic_linear(hidden_critic)
        
        return value, action_log_probs, dist_entropy, states, action_probs

    def get_bonus(self, eta, states, next_states, actions, action_probs):
        action_pred, phi2_pred, phi1, phi2 =  self.icm(states, next_states, action_probs)
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2, reduce=False).sum(-1).unsqueeze(-1)
        return eta * forward_loss

    def get_icm_loss(self, states, next_states, actions, action_probs):
        action_pred, phi2_pred, phi1, phi2 =  self.icm(states, next_states, action_probs)
        inverse_loss = F.cross_entropy(action_pred, actions.view(-1))
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2.detach(), reduce=False).sum(-1).mean()
        return inverse_loss, forward_loss


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_gru, use_icm):
        super(CNNPolicy, self).__init__()
        self.head = NatureHead(num_inputs)

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
        if use_icm:
            self.icm = ICM(action_space, 512, num_inputs)

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)
        self.head.apply(weights_init_head)
        
        if hasattr(self, 'icm'):
            self.icm.head.apply(weights_init_head)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.head(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return x, x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_icm):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        if use_icm:
            self.icm = ICM(action_space, num_inputs, cnn_head=False)
        
        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)

        self.critic_linear = nn.Linear(64, 1)
        self.dist = get_distribution(64, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        hidden_critic = F.tanh(x)

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        hidden_actor = F.tanh(x)

        return hidden_critic, hidden_actor, states
