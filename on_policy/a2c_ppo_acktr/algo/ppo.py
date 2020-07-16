import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 value_loss_coef,
                 entropy_coef,
                 direction_loss_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.direction_loss_coef = direction_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
    
    def _update(self, generator, postwidth=0):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        direction_loss_epoch = 0
        cnt = 0
        
        for sample in generator:
            obs_batch, \
            actions_batch, \
            value_preds_batch, \
            return_batch, \
            old_action_log_probs_batch, \
            adv_targ, \
            target_pcs = sample

            values, \
            action_log_probs, \
            dist_entropy, \
            dist_mode = self.actor_critic.evaluate_actions(obs_batch, actions_batch)
            dist_mode = dist_mode.view(actions_batch.shape)

            ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            direction_loss = torch.Tensor([0]).float()
            if len(target_pcs) > 2:
                print(len(target_pcs))
            for i, target_pc in enumerate(target_pcs):
                if target_pc is None:
                    continue
                direction_loss += adv_targ * ((dist_mode[i, 1:postwidth+1] - target_pc)**2).mean()
            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (return_batch - values).pow(2).mean()
                

            self.optimizer.zero_grad()
            loss = value_loss * self.value_loss_coef + \
                   action_loss + \
                   direction_loss * self.direction_loss_coef - \
                   dist_entropy * self.entropy_coef
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()
            direction_loss_epoch += direction_loss.item()
            cnt += 1
        
        value_loss_epoch /= cnt
        action_loss_epoch /= cnt
        dist_entropy_epoch /= cnt
        direction_loss_epoch /= cnt

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, direction_loss_epoch

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages)

            value_loss, action_loss, dist_entropy = self._update(data_generator)
            value_loss_epoch += value_loss
            action_loss_epoch += action_loss
            dist_entropy_epoch += dist_entropy

        num_updates = self.ppo_epoch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
