import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        transition,
        decoder,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        use_failure_head=False,
        failure_head_hidden_dim=256,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.transition = transition
        self.decoder = decoder
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * concat_dim

        # Failure head: mean-pool predictor output patches → scalar safety score
        # Input dim equals the predictor output dim (emb_dim when concat_dim=1)
        self.use_failure_head = use_failure_head
        if use_failure_head:
            self.failure_head = nn.Sequential(
                nn.LayerNorm(self.emb_dim),
                nn.Linear(self.emb_dim, failure_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(failure_head_hidden_dim, 1),
            )

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16
            num_side = image_size // decoder_scale
            self.encoder_image_size = num_side * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.train_predictor:
            self.transition.train(mode)
        if self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.transition.eval()
        self.decoder.eval()

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        z = self.transition.build_z(z_dct["visual"], z_dct["proprio"], act)
        return z

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual" (b, t, num_patches, encoder_emb_dim), "proprio" (b, t, proprio_dim) raw
        """
        visual = obs["visual"]
        b, t = visual.shape[0], visual.shape[1]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        enc_out = self.encoder.forward(visual)
        visual_embs = enc_out["patch_tokens"]
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b, t=t)
        class_token = rearrange(enc_out["class_token"], "(b t) d -> b t d", b=b, t=t)

        proprio = obs["proprio"]  # raw, transition will embed
        return {"visual": visual_embs, "proprio": proprio, "class_token": class_token}

    def predict_failure(self, z):
        """Predict a per-timestep safety score from predictor latents.

        Mirrors the reference failure_head: mean-pool across patches, then MLP → scalar.

        Args:
            z: (B, T, num_patches, predictor_dim)  — output of predict() or encode()
        Returns:
            scores: (B, T, 1)  — positive = safe, negative = unsafe (before tanh scaling)
        """
        assert self.use_failure_head, "failure_head not enabled (use_failure_head=False)"
        pooled = z.mean(dim=2)          # (B, T, predictor_dim)
        return self.failure_head(pooled)  # (B, T, 1)

    def predict(self, z):
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        return self.transition.forward_z(z)

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs = self.decode_obs(z_obs)
        return obs

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual = self.decoder(z_obs["visual"])
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"],
        }
        return obs

    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = (
                z[..., : -(self.proprio_dim + self.action_dim)],
                z[..., -(self.proprio_dim + self.action_dim) : -self.action_dim],
                z[..., -self.action_dim :],
            )
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]
        visual_tgt = obs["visual"][:, self.num_pred :, ...]

        z_pred = self.predict(z_src)
        obs_pred = self.decode(z_pred.detach())
        visual_pred = obs_pred["visual"]
        recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
        loss_components["decoder_recon_loss_pred"] = recon_loss_pred
        loss_components["decoder_loss_pred"] = recon_loss_pred

        if self.concat_dim == 0:
            z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
            z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
            z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
        elif self.concat_dim == 1:
            z_visual_loss = self.emb_criterion(
                z_pred[:, :, :, : -(self.proprio_dim + self.action_dim)],
                z_tgt[:, :, :, : -(self.proprio_dim + self.action_dim)].detach(),
            )
            z_proprio_loss = self.emb_criterion(
                z_pred[:, :, :, -(self.proprio_dim + self.action_dim) : -self.action_dim],
                z_tgt[:, :, :, -(self.proprio_dim + self.action_dim) : -self.action_dim].detach(),
            )
            z_loss = self.emb_criterion(
                z_pred[:, :, :, : -self.action_dim],
                z_tgt[:, :, :, : -self.action_dim].detach(),
            )

        loss = loss + z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss

        obs_reconstructed = self.decode(z.detach())
        visual_reconstructed = obs_reconstructed["visual"]
        recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs["visual"])

        loss_components["decoder_recon_loss_reconstructed"] = recon_loss_reconstructed
        loss_components["decoder_loss_reconstructed"] = recon_loss_reconstructed
        loss = loss + recon_loss_reconstructed
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        return self.transition.replace_actions_in_z(z, act)

    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0["visual"].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1:, ...]
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
