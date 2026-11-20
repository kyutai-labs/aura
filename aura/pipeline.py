"""AURA streaming inference: reference image + speech -> talking-head video, chunk by chunk."""

import itertools
import logging
import time

import torch
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors.torch import load_file

from .audio import MimiFeatures, Wav2Vec2Features, to_video_rate
from .configs import CONFIGS, ModelConfig
from .kv_cache import KVCache
from .model import AuraDiT
from .wan.t5 import T5EncoderModel
from .wan.vae2_1 import Wan2_1_VAE
from .wan.vae2_2 import Wan2_2_VAE, unpatchify

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "A person is talking to the camera, with natural facial expressions and head movements."


class FlowSchedule:
    """Noise levels of the flow-matching schedule the students were distilled on (50 steps,
    shift 5); a timestep maps to the sigma of the nearest scheduler step."""

    def __init__(self, steps=50, shift=5.0, sigma_min=0.003 / 1.002):
        sigmas = torch.linspace(1.0, sigma_min, steps)
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * 1000

    def sigma(self, t: float) -> float:
        return float(self.sigmas[(self.timesteps - t).abs().argmin()].double())


class Aura:
    def __init__(self, name: str = "aura-1.3b", device: str = "cuda", compile: bool = False):
        self.cfg: ModelConfig = CONFIGS[name]
        self.device = torch.device(device)
        cfg = self.cfg
        wan_dir = snapshot_download(
            cfg.wan_repo, allow_patterns=[cfg.vae, "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]
        )

        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=f"{wan_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=f"{wan_dir}/google/umt5-xxl",
        )
        vae_cls = Wan2_1_VAE if cfg.vae == "Wan2.1_VAE.pth" else Wan2_2_VAE
        self.vae = vae_cls(vae_pth=f"{wan_dir}/{cfg.vae}", dtype=cfg.vae_dtype, device=self.device)

        with torch.device("meta"):
            self.model = AuraDiT(cfg)
        state = load_file(hf_hub_download(cfg.repo, cfg.checkpoint))
        state = {k: v for k, v in state.items() if not k.startswith("frame_packer.")}  # unused when streaming
        state["trainable_cond_mask.weight"] = state["trainable_cond_mask.weight"][:2]
        self.model.load_state_dict(state, assign=True)
        self.model.eval().requires_grad_(False).to(self.device)
        if compile:
            for i, block in enumerate(self.model.blocks):
                self.model.blocks[i] = torch.compile(block, dynamic=True)

        audio_cls = MimiFeatures if cfg.audio_backend == "mimi" else Wav2Vec2Features
        self.audio_features = audio_cls(self.device)
        self.schedule = FlowSchedule()

    @property
    def fps(self) -> int:
        return self.cfg.fps

    def load_image(self, path: str) -> torch.Tensor:
        """Resize and center-crop to the closest training resolution -> [1, 3, 1, H, W] in [-1, 1]."""
        image = Image.open(path).convert("RGB")
        ratio = image.height / image.width
        h, w = self.cfg.buckets[min(self.cfg.buckets, key=lambda k: abs(k - ratio))]
        image = TF.center_crop(TF.resize(image, min(h, w)), [h, w])
        return (TF.to_tensor(image) * 2 - 1)[None, :, None].to(self.device)

    def decode(self, latents: torch.Tensor, first: bool) -> torch.Tensor:
        """Decode one chunk [C, F, h, w] with the causal VAE, continuing from the previous chunk."""
        vae = self.vae.model
        mean, inv_std = self.vae.scale
        is_v22 = isinstance(self.vae, Wan2_2_VAE)
        with torch.autocast("cuda", dtype=self.cfg.vae_dtype):
            z = vae.conv2(latents[None] / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1))
            t, group = z.shape[2], self.cfg.vae_decode_group
            # The first latent frame of the video decodes alone, into a single frame.
            edges = [0, *range(1, t, group), t] if first else [*range(0, t, group), t]
            out = []
            for start, end in itertools.pairwise(edges):
                vae._conv_idx = [0]
                kwargs = {"first_chunk": True} if is_v22 and first and start == 0 else {}
                out.append(vae.decoder(z[:, :, start:end], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kwargs))
            video = torch.cat(out, dim=2).float().clamp_(-1, 1)
        return unpatchify(video, patch_size=2)[0] if is_v22 else video[0]

    @torch.no_grad()
    def generate(
        self, image: str, audio: str, prompt: str = DEFAULT_PROMPT, num_chunks: int | None = None, seed: int = 42
    ):
        """Yield the video chunk by chunk, as [3, frames, H, W] tensors in [-1, 1].

        Each chunk is `frames_per_block` latent frames (16 video frames, 13 for the first chunk).
        The video covers the whole audio, rounded up to a whole chunk, or `num_chunks` chunks.
        """
        cfg = self.cfg
        frames_per_chunk = cfg.vae_stride[0] * cfg.frames_per_block
        features, n = to_video_rate(self.audio_features(audio), cfg.fps, frames_per_chunk)
        features = features.to(self.device, torch.bfloat16)
        num_chunks = n if num_chunks is None else min(num_chunks, n)

        ref = self.load_image(image)
        ref_latent = torch.stack(self.vae.encode(list(ref.to(cfg.vae_dtype))))
        context = self.text_encoder([prompt], self.device)[0]

        _, _, _, height, width = ref.shape
        lat_h, lat_w = height // cfg.vae_stride[1], width // cfg.vae_stride[2]
        tokens_per_frame = (lat_h // cfg.patch_size[1]) * (lat_w // cfg.patch_size[2])
        kv_cache = KVCache(
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.dim // cfg.num_heads,
            tokens_per_frame=tokens_per_frame,
            frames_per_block=cfg.frames_per_block,
            window=cfg.kv_window,
            segment_frames=cfg.segment_frames,
            ref_rope_offset=cfg.ref_rope_offset,
            # Last block that fits in a training clip.
            rope_max_block=cfg.segment_frames // cfg.frames_per_block - 1 if cfg.rope_reindex else None,
            device=self.device,
        )
        self.vae.model.clear_cache()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for r in range(num_chunks):
                t0 = time.perf_counter()
                generator = torch.Generator(self.device).manual_seed(seed + r)
                shape = (cfg.in_dim, cfg.frames_per_block, lat_h, lat_w)
                x = torch.randn(shape, dtype=torch.bfloat16, device=self.device, generator=generator)
                audio_tokens = self.model.encode_audio(features[..., r * frames_per_chunk : (r + 1) * frames_per_chunk])

                # Few-step sampling: predict the clean latents, re-noise them to the next level.
                steps = cfg.denoising_steps
                for i, t in enumerate(steps):
                    timestep = torch.full((1,), t, dtype=torch.long, device=self.device)
                    velocity = self.model(x, timestep, context, ref_latent, audio_tokens, kv_cache)
                    x0 = (x.double() - self.schedule.sigma(t) * velocity.double()).float()
                    if i + 1 < len(steps):
                        sigma = self.schedule.sigma(steps[i + 1])
                        noise = torch.randn(x0.shape, dtype=x0.dtype, device=self.device, generator=generator)
                        x = (1 - sigma) * x0 + sigma * noise
                kv_cache.advance()

                video = self.decode(x0, first=r == 0)
                torch.cuda.synchronize()
                logger.info(f"chunk {r + 1}/{num_chunks}: {video.shape[1]} frames in {time.perf_counter() - t0:.2f}s")
                yield video.cpu()
        self.vae.model.clear_cache()
