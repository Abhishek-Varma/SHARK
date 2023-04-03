from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel
from collections import defaultdict
import torch
import safetensors.torch
import traceback
import sys
import os
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    get_opt_flags,
    base_models,
    args,
    fetch_or_delete_vmfbs,
    preprocessCKPT,
    get_path_to_diffusers_checkpoint,
    fetch_and_update_base_model_id,
    get_path_stem,
    get_extended_name,
    get_stencil_model_id,
    update_lora_weight,
)

##################################################################################################
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.
    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.half)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=idx_buffer_view.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])
        # If the batch size is odd, then it's not possible for promted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, not use_rand)
    else:
        m, u = (do_nothing, do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good







def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock






def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock

def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).
    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    # is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    # if not is_diffusers:
    #     if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
    #         # Provided model not supported
    #         raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
    #     diffusion_model = model.model.diffusion_model
    # else:
    #     # Supports "pipe.unet" and "unet"
    #     diffusion_model = model.unet if hasattr(model, "unet") else model
    diffusion_model = model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if False else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not False:
                module.disable_self_attn = False

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return 
##################################################################################################
# These shapes are parameter dependent.
def replace_shape_str(shape, max_len, width, height, batch_size):
    new_shape = []
    for i in range(len(shape)):
        if shape[i] == "max_len":
            new_shape.append(max_len)
        elif shape[i] == "height":
            new_shape.append(height)
        elif shape[i] == "width":
            new_shape.append(width)
        elif isinstance(shape[i], str):
            if "*" in shape[i]:
                mul_val = int(shape[i].split("*")[0])
                if "batch_size" in shape[i]:
                    new_shape.append(batch_size * mul_val)
                elif "height" in shape[i]:
                    new_shape.append(height * mul_val)
                elif "width" in shape[i]:
                    new_shape.append(width * mul_val)
            elif "/" in shape[i]:
                import math
                div_val = int(shape[i].split("/")[1])
                if "batch_size" in shape[i]:
                    new_shape.append(math.ceil(batch_size / div_val))
                elif "height" in shape[i]:
                    new_shape.append(math.ceil(height / div_val))
                elif "width" in shape[i]:
                    new_shape.append(math.ceil(width / div_val))
        else:
            new_shape.append(shape[i])
    return new_shape


# Get the input info for various models i.e. "unet", "clip", "vae", "vae_encode".
def get_input_info(model_info, max_len, width, height, batch_size):
    dtype_config = {"f32": torch.float32, "i64": torch.int64}
    input_map = defaultdict(list)
    for k in model_info:
        for inp in model_info[k]:
            shape = model_info[k][inp]["shape"]
            dtype = dtype_config[model_info[k][inp]["dtype"]]
            tensor = None
            if isinstance(shape, list):
                clean_shape = replace_shape_str(
                    shape, max_len, width, height, batch_size
                )
                if dtype == torch.int64:
                    tensor = torch.randint(1, 3, tuple(clean_shape))
                else:
                    tensor = torch.randn(*clean_shape).to(dtype)
            elif isinstance(shape, int):
                tensor = torch.tensor(shape).to(dtype)
            else:
                sys.exit("shape isn't specified correctly.")
            input_map[k].append(tensor)
    return input_map


class SharkifyStableDiffusionModel:
    def __init__(
        self,
        model_id: str,
        custom_weights: str,
        custom_vae: str,
        precision: str,
        max_len: int = 64,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        use_base_vae: bool = False,
        use_tuned: bool = False,
        low_cpu_mem_usage: bool = False,
        debug: bool = False,
        sharktank_dir: str = "",
        generate_vmfb: bool = True,
        is_inpaint: bool = False,
        is_upscaler: bool = False,
        use_stencil: str = None,
        use_lora: str = "",
        use_quantize: str = None,
    ):
        self.check_params(max_len, width, height)
        self.max_len = max_len
        self.height = height // 8
        self.width = width // 8
        self.batch_size = batch_size
        self.custom_weights = custom_weights
        self.use_quantize = use_quantize
        if custom_weights != "":
            assert custom_weights.lower().endswith(
                (".ckpt", ".safetensors")
            ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
            custom_weights = get_path_to_diffusers_checkpoint(custom_weights)
        self.model_id = model_id if custom_weights == "" else custom_weights
        # TODO: remove the following line when stable-diffusion-2-1 works
        if self.model_id == "stabilityai/stable-diffusion-2-1":
            self.model_id = "stabilityai/stable-diffusion-2-1-base"
        self.custom_vae = custom_vae
        self.precision = precision
        self.base_vae = use_base_vae
        self.model_name = (
            "_trick"
            + str(batch_size)
            + "_"
            + str(max_len)
            + "_"
            + str(height)
            + "_"
            + str(width)
            + "_"
            + precision
        )
        print(f'use_tuned? sharkify: {use_tuned}')
        self.use_tuned = use_tuned
        if use_tuned:
            self.model_name = self.model_name + "_tuned"
        self.model_name = self.model_name + "_" + get_path_stem(self.model_id)
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.is_inpaint = is_inpaint
        self.is_upscaler = is_upscaler
        self.use_stencil = get_stencil_model_id(use_stencil)
        if use_lora != "":
            self.model_name = self.model_name + "_" + get_path_stem(use_lora)
        self.use_lora = use_lora

        print(self.model_name)
        self.debug = debug
        self.sharktank_dir = sharktank_dir
        self.generate_vmfb = generate_vmfb

    def get_extended_name_for_all_model(self, mask_to_fetch):
        model_name = {}
        sub_model_list = ["clip", "unet", "stencil_unet", "vae", "vae_encode", "stencil_adaptor"]
        index = 0
        for model in sub_model_list:
            if mask_to_fetch[index] == False:
                index += 1
                continue
            sub_model = model
            model_config = self.model_name
            if "vae" == model:
                if self.custom_vae != "":
                    model_config = model_config + get_path_stem(self.custom_vae)
                if self.base_vae:
                    sub_model = "base_vae"
            model_name[model] = get_extended_name(sub_model + model_config)
            index += 1
        return model_name

    def check_params(self, max_len, width, height):
        if not (max_len >= 32 and max_len <= 77):
            sys.exit("please specify max_len in the range [32, 77].")
        if not (width % 8 == 0 and width >= 128):
            sys.exit("width should be greater than 128 and multiple of 8")
        if not (height % 8 == 0 and height >= 128):
            sys.exit("height should be greater than 128 and multiple of 8")

    def get_vae_encode(self):
        class VaeEncodeModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, low_cpu_mem_usage=False):
                super().__init__()
                self.vae = AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )

            def forward(self, input):
                latents = self.vae.encode(input).latent_dist.sample()
                return 0.18215 * latents

        vae_encode = VaeEncodeModel()
        inputs = tuple(self.inputs["vae_encode"])
        is_f16 = True if self.precision == "fp16" else False
        shark_vae_encode = compile_through_fx(
            vae_encode,
            inputs,
            is_f16=is_f16,
            use_tuned=self.use_tuned,
            model_name=self.model_name["vae_encode"],
            extra_args=get_opt_flags("vae", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_vae_encode

    def get_vae(self):
        class VaeModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, base_vae=self.base_vae, custom_vae=self.custom_vae, low_cpu_mem_usage=False):
                super().__init__()
                self.vae = None
                if custom_vae == "":
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                elif not isinstance(custom_vae, dict):
                    self.vae = AutoencoderKL.from_pretrained(
                        custom_vae,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                else:
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.vae.load_state_dict(custom_vae)
                self.base_vae = base_vae

            def forward(self, input):
                if not self.base_vae:
                    input = 1 / 0.18215 * input
                x = self.vae.decode(input, return_dict=False)[0]
                x = (x / 2 + 0.5).clamp(0, 1)
                if self.base_vae:
                    return x
                x = x * 255.0
                return x.round()

        vae = VaeModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        inputs = tuple(self.inputs["vae"])
        is_f16 = True if self.precision == "fp16" else False
        save_dir = os.path.join(self.sharktank_dir, self.model_name["vae"])
        if self.debug:
            os.makedirs(save_dir, exist_ok=True)
        shark_vae = compile_through_fx(
            vae,
            inputs,
            is_f16=is_f16,
            use_tuned=self.use_tuned,
            model_name=self.model_name["vae"],
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("vae", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_vae

    def get_vae_upscaler(self):
        class VaeModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, low_cpu_mem_usage=False):
                super().__init__()
                self.vae = AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )

            def forward(self, input):
                x = self.vae.decode(input, return_dict=False)[0]
                x = (x / 2 + 0.5).clamp(0, 1)
                return x

        vae = VaeModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        inputs = tuple(self.inputs["vae"])
        shark_vae = compile_through_fx(
            vae,
            inputs,
            use_tuned=self.use_tuned,
            model_name=self.model_name["vae"],
            extra_args=get_opt_flags("vae", precision="fp32"),
            base_model_id=self.base_model_id,
        )
        return shark_vae

    def get_controlled_unet(self):
        class ControlledUnetModel(torch.nn.Module):
            def __init__(
                self, model_id=self.model_id, low_cpu_mem_usage=False, use_lora=self.use_lora
            ):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                if use_lora != "":
                    update_lora_weight(self.unet, use_lora, "unet")
                self.in_channels = self.unet.in_channels
                self.train(False)

            def forward( self, latent, timestep, text_embedding, guidance_scale, control1,
                         control2, control3, control4, control5, control6, control7,
                         control8, control9, control10, control11, control12, control13,
            ):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                db_res_samples = tuple([ control1, control2, control3, control4, control5, control6, control7, control8, control9, control10, control11, control12,])
                mb_res_samples = control13
                latents = torch.cat([latent] * 2)
                unet_out = self.unet.forward(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embedding,
                    down_block_additional_residuals=db_res_samples,
                    mid_block_additional_residual=mb_res_samples,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                return noise_pred

        unet = ControlledUnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False

        inputs = tuple(self.inputs["stencil_unet"])
        input_mask = [True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True,]
        shark_controlled_unet = compile_through_fx(
            unet,
            inputs,
            model_name=self.model_name["stencil_unet"],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_controlled_unet

    def get_control_net(self):
        class StencilControlNetModel(torch.nn.Module):
            def __init__(
                self, model_id=self.use_stencil, low_cpu_mem_usage=False
            ):
                super().__init__()
                self.cnet = ControlNetModel.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self.in_channels = self.cnet.in_channels
                self.train(False)

            def forward(
                self,
                latent,
                timestep,
                text_embedding,
                stencil_image_input,
            ):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # TODO: guidance NOT NEEDED change in `get_input_info` later
                latents = torch.cat(
                    [latent] * 2
                )  # needs to be same as controlledUNET latents
                stencil_image = torch.cat(
                    [stencil_image_input] * 2
                )  # needs to be same as controlledUNET latents
                down_block_res_samples, mid_block_res_sample = self.cnet.forward(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embedding,
                    controlnet_cond=stencil_image,
                    return_dict=False,
                )
                return tuple(list(down_block_res_samples) + [mid_block_res_sample])

        scnet = StencilControlNetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False

        inputs = tuple(self.inputs["stencil_adaptor"])
        input_mask = [True, True, True, True]
        shark_cnet = compile_through_fx(
            scnet,
            inputs,
            model_name=self.model_name["stencil_adaptor"],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_cnet

    def get_unet(self):
        import faulthandler
        faulthandler.enable()
        class UnetModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, low_cpu_mem_usage=False, use_lora=self.use_lora):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self.unet = apply_patch(self.unet, ratio=0.5)
                # print("========AFTER==========")
                # for _, module in self.unet.named_modules():
                #     if isinstance_str(module, "ToMeBlock"):
                #         print("--A--",str(module))
                if use_lora != "":
                    update_lora_weight(self.unet, use_lora, "unet")
                self.in_channels = self.unet.in_channels
                self.train(False)
                if(args.attention_slicing is not None and args.attention_slicing != "none"):
                    if(args.attention_slicing.isdigit()):
                        self.unet.set_attention_slice(int(args.attention_slicing))
                    else:
                        self.unet.set_attention_slice(args.attention_slicing)

            # TODO: Instead of flattening the `control` try to use the list.
            def forward(
                self, latent, timestep, text_embedding, guidance_scale,
            ):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latents = torch.cat([latent] * 2)
                unet_out = self.unet.forward(
                    latents, timestep, text_embedding, return_dict=False
                )[0]
                noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                return noise_pred

        unet = UnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        input_mask = [True, True, True, False]
        save_dir = os.path.join(self.sharktank_dir, self.model_name["unet"])
        if self.debug:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        shark_unet = compile_through_fx(
            unet,
            inputs,
            model_name=self.model_name["unet"],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_unet

    def get_unet_upscaler(self):
        class UnetModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, low_cpu_mem_usage=False):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self.in_channels = self.unet.in_channels
                self.train(False)

            def forward(self, latent, timestep, text_embedding, noise_level):
                unet_out = self.unet.forward(
                    latent,
                    timestep,
                    text_embedding,
                    noise_level,
                    return_dict=False,
                )[0]
                return unet_out

        unet = UnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        input_mask = [True, True, True, False]
        shark_unet = compile_through_fx(
            unet,
            inputs,
            model_name=self.model_name["unet"],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
        )
        return shark_unet

    def get_clip(self):
        class CLIPText(torch.nn.Module):
            def __init__(self, model_id=self.model_id, low_cpu_mem_usage=False, use_lora=self.use_lora):
                super().__init__()
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                if use_lora != "":
                    update_lora_weight(self.text_encoder, use_lora, "text_encoder")

            def forward(self, input):
                return self.text_encoder(input)[0]

        clip_model = CLIPText(low_cpu_mem_usage=self.low_cpu_mem_usage)
        save_dir = os.path.join(self.sharktank_dir, self.model_name["clip"])
        if self.debug:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        shark_clip = compile_through_fx(
            clip_model,
            tuple(self.inputs["clip"]),
            model_name=self.model_name["clip"],
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("clip", precision="fp32"),
            base_model_id=self.base_model_id,
        )
        return shark_clip

    def process_custom_vae(self):
        custom_vae = self.custom_vae.lower()
        if not custom_vae.endswith((".ckpt", ".safetensors")):
            return self.custom_vae
        try:
            preprocessCKPT(self.custom_vae)
            return get_path_to_diffusers_checkpoint(self.custom_vae)
        except:
            print("Processing standalone Vae checkpoint")
            vae_checkpoint = None
            vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
            if custom_vae.endswith(".ckpt"):
                vae_checkpoint = torch.load(self.custom_vae, map_location="cpu")
            else:
                vae_checkpoint = safetensors.torch.load_file(self.custom_vae, device="cpu")
            if "state_dict" in vae_checkpoint:
                vae_checkpoint = vae_checkpoint["state_dict"]
            vae_dict = {k: v for k, v in vae_checkpoint.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
            return vae_dict
        
            
    # Compiles Clip, Unet and Vae with `base_model_id` as defining their input
    # configiration.
    def compile_all(self, base_model_id, need_vae_encode, need_stencil):
        self.base_model_id = base_model_id
        self.inputs = get_input_info(
            base_models[base_model_id],
            self.max_len,
            self.width,
            self.height,
            self.batch_size,
        )
        if self.is_upscaler:
            return self.get_clip(), self.get_unet_upscaler(), self.get_vae_upscaler()

        compiled_controlnet = None
        compiled_controlled_unet = None
        compiled_unet = None
        if need_stencil:
            compiled_controlnet = self.get_control_net()
            compiled_controlled_unet = self.get_controlled_unet()
        else:
            # TODO: Plug the experimental "int8" support at right place.
            if self.use_quantize == "int8":
                from apps.stable_diffusion.src.models.opt_params import get_unet
                compiled_unet = get_unet()
            else:
                compiled_unet = self.get_unet()
        if self.custom_vae != "":
            print("Plugging in custom Vae")
        compiled_vae = self.get_vae()
        compiled_clip = self.get_clip()

        if need_stencil:
            return compiled_clip, compiled_controlled_unet, compiled_vae, compiled_controlnet
        if need_vae_encode:
            compiled_vae_encode = self.get_vae_encode()
            return compiled_clip, compiled_unet, compiled_vae, compiled_vae_encode

        return compiled_clip, compiled_unet, compiled_vae

    def __call__(self):
        # Step 1:
        # --  Fetch all vmfbs for the model, if present, else delete the lot.
        need_vae_encode, need_stencil = False, False
        if not self.is_upscaler and args.img_path is not None:
            if self.use_stencil is not None:
                need_stencil = True
            else:
                need_vae_encode = True
        # `mask_to_fetch` prepares a mask to pick a combination out of :-
        # ["clip", "unet", "stencil_unet", "vae", "vae_encode", "stencil_adaptor"]
        mask_to_fetch = [True, True, False, True, False, False]
        if need_vae_encode:
            mask_to_fetch = [True, True, False, True, True, False]
        elif need_stencil:
            mask_to_fetch = [True, False, True, True, False, True]
        self.model_name = self.get_extended_name_for_all_model(mask_to_fetch)
        vmfbs = fetch_or_delete_vmfbs(self.model_name, self.precision)   
        if vmfbs[0]:
            # -- If all vmfbs are indeed present, we also try and fetch the base
            #    model configuration for running SD with custom checkpoints.
            if self.custom_weights != "":
                args.hf_model_id = fetch_and_update_base_model_id(self.custom_weights)
            if args.hf_model_id == "":
                sys.exit("Base model configuration for the custom model is missing. Use `--clear_all` and re-run.")
            print("Loaded vmfbs from cache and successfully fetched base model configuration.")
            return vmfbs

        # Step 2:
        # -- If vmfbs weren't found, we try to see if the base model configuration
        #    for the required SD run is known to us and bypass the retry mechanism.
        model_to_run = ""
        if self.custom_weights != "":
            model_to_run = self.custom_weights
            assert self.custom_weights.lower().endswith(
                (".ckpt", ".safetensors")
            ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
            preprocessCKPT(self.custom_weights, self.is_inpaint)
        else:
            model_to_run = args.hf_model_id
        # For custom Vae user can provide either the repo-id or a checkpoint file,
        # and for a checkpoint file we'd need to process it via Diffusers' script.
        self.custom_vae = self.process_custom_vae()
        base_model_fetched = fetch_and_update_base_model_id(model_to_run)
        if base_model_fetched != "":
            print("Compiling all the models with the fetched base model configuration.")
            if args.ckpt_loc != "":
                args.hf_model_id = base_model_fetched
            return self.compile_all(base_model_fetched, need_vae_encode, need_stencil)

        # Step 3:
        # -- This is the retry mechanism where the base model's configuration is not
        #    known to us and figure that out by trial and error.
        print("Inferring base model configuration.")
        for model_id in base_models:
            try:
                if need_vae_encode:
                    compiled_clip, compiled_unet, compiled_vae, compiled_vae_encode = self.compile_all(model_id, need_vae_encode, need_stencil)
                elif need_stencil:
                    compiled_clip, compiled_unet, compiled_vae, compiled_controlnet = self.compile_all(model_id, need_vae_encode, need_stencil)
                else:
                    compiled_clip, compiled_unet, compiled_vae = self.compile_all(model_id, need_vae_encode, need_stencil)
            except Exception as e:
                print(e)
                print("Retrying with a different base model configuration")
                continue
            # -- Once a successful compilation has taken place we'd want to store
            #    the base model's configuration inferred.
            fetch_and_update_base_model_id(model_to_run, model_id)
            # This is done just because in main.py we are basing the choice of tokenizer and scheduler
            # on `args.hf_model_id`. Since now, we don't maintain 1:1 mapping of variants and the base
            # model and rely on retrying method to find the input configuration, we should also update
            # the knowledge of base model id accordingly into `args.hf_model_id`.
            if args.ckpt_loc != "":
                args.hf_model_id = model_id
            if need_vae_encode:
                return (
                    compiled_clip,
                    compiled_unet,
                    compiled_vae,
                    compiled_vae_encode,
                )
            if need_stencil:
                return (
                    compiled_clip,
                    compiled_unet,
                    compiled_vae,
                    compiled_controlnet,
                )
            return compiled_clip, compiled_unet, compiled_vae
        sys.exit(
            "Cannot compile the model. Please create an issue with the detailed log at https://github.com/nod-ai/SHARK/issues"
        )
