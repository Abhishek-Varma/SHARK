from apps.language_models.src.model_wrappers.minigpt4 import (
    LayerNorm,
    VisionModel,
    QformerBertModel,
    FirstLlamaModel,
    SecondLlamaModel,
    StoppingCriteriaSub,
    CONV_VISION,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from omegaconf import OmegaConf
from pathlib import Path
from shark.shark_downloader import download_public_file
from shark.shark_inference import SharkInference
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import StoppingCriteriaList
from transformers.generation import GenerationConfig, LogitsProcessorList
import copy

import re
import torch
import os
from PIL import Image

import os
import torch
import sys

# SHARK dependencies
# from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    args,
)
import random
import contextlib
import re
from transformers import BertTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.generation import GenerationConfig, LogitsProcessorList
import copy
# QFormer, eva_vit, blip_processor, dist_utils
from apps.language_models.src.pipelines.minigpt4_utils.Qformer import BertConfig, BertLMHeadModel
from apps.language_models.src.pipelines.minigpt4_utils.dist_utils import download_cached_file
from apps.language_models.src.pipelines.minigpt4_utils.eva_vit import create_eva_vit_g
from apps.language_models.src.pipelines.minigpt4_utils.blip_processors import Blip2ImageEvalProcessor

import argparse

parser = argparse.ArgumentParser(
    prog="MiniGPT4 runner",
    description="runs MiniGPT4",
)

parser.add_argument(
    "--precision", "-p", default="fp16", help="fp32, fp16, int8, int4"
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--vision_model_vmfb_path", default=None, help="path to vision model's vmfb"
)
parser.add_argument(
    "--vision_model_mlir_path",
    default=None,
    help="path to vision model's mlir file",
)
parser.add_argument(
    "--qformer_bert_model_vmfb_path", default=None, help="path to qformer model's vmfb"
)
parser.add_argument(
    "--qformer_bert_model_mlir_path",
    default=None,
    help="path to qformer model's mlir file",
)
parser.add_argument(
    "--first_llama_vmfb_path", default=None, help="path to first llama model's vmfb"
)
parser.add_argument(
    "--first_llama_mlir_path",
    default=None,
    help="path to first llama model's mlir file",
)
parser.add_argument(
    "--second_llama_vmfb_path", default=None, help="path to second llama model's vmfb"
)
parser.add_argument(
    "--second_llama_mlir_path",
    default=None,
    help="path to second llama model's mlir file",
)
parser.add_argument(
    "--image_path",
    type=str,
    default="",
    help="path to the input image",
)
parser.add_argument(
    "--load_mlir_from_shark_tank",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="download precompile mlir from shark tank",
)
parser.add_argument(
    "--cli",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Run model in cli mode",
)
parser.add_argument(
    "--compile",
    # Change this to True when raising PR.
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Compile all models",
)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url

class MiniGPT4BaseModel(torch.nn.Module):
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/minigpt4.yaml",
    }
    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision
    
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        self.load_state_dict(state_dict, strict=False)
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        print("vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision")
        print(vit_model, " ", img_size, " ",drop_path_rate," ", use_grad_checkpoint, " ",vit_precision)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for _, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            # logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        print("num_query_token, self.visual_encoder.num_features")
        print(num_query_token, " ", self.visual_encoder.num_features)
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for _, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            # logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print("Llama = ", llama_model)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float32,
            )

        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = torch.nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

class MiniGPT4(SharkLLMBase):
    def __init__(
        self, model_name, hf_model_path=None, max_new_tokens=300,
        device="cuda", precision="fp16", _compile=False,
        vision_model_mlir_path=Path("minigpt4_vision_model_fp16.mlir"),
        vision_model_vmfb_path=Path("minigpt4_vision_model_fp16_cuda.vmfb"),
        qformer_bert_model_mlir_path = Path("minigpt4_qformer_bert_model_fp32.mlir"),
        qformer_bert_model_vmfb_path = Path("minigpt4_qformer_bert_model_fp32_cuda.vmfb"),
        first_llama_mlir_path = Path("minigpt4_first_llama_fp16.mlir"),
        first_llama_vmfb_path = Path("minigpt4_first_llama_fp16_cuda.vmfb"),
        second_llama_mlir_path = Path("minigpt4_second_llama_fp16.mlir"),
        second_llama_vmfb_path = Path("minigpt4_second_llama_fp16_cuda.vmfb"),
    ) -> None:
        self.model_name = model_name
        self.shark_model = None
        super().__init__(model_name, hf_model_path, max_new_tokens)

        self.device = device
        self.precision = precision
        self._compile = _compile

        self.vision_model_mlir_path = vision_model_mlir_path
        self.qformer_bert_model_mlir_path = qformer_bert_model_mlir_path
        self.first_llama_mlir_path = first_llama_mlir_path
        self.second_llama_mlir_path = second_llama_mlir_path
        
        self.vision_model_vmfb_path = vision_model_vmfb_path
        self.qformer_bert_model_vmfb_path = qformer_bert_model_vmfb_path
        self.first_llama_vmfb_path = first_llama_vmfb_path
        self.second_llama_vmfb_path = second_llama_vmfb_path

        print('Initializing Chat')
        config = OmegaConf.load("apps/language_models/src/pipelines/minigpt4_utils/configs/minigpt4_eval.yaml")
        model_config = OmegaConf.create()
        model_config = OmegaConf.merge(
            model_config,
            OmegaConf.load('apps/language_models/src/pipelines/minigpt4_utils/configs/minigpt4.yaml'),
            {"model": config["model"]},
        )
        model_config = model_config['model']
        model_config.device_8bit = 0
        model = MiniGPT4BaseModel.from_config(model_config).to('cpu')
        datasets = config.get("datasets", None)
        dataset_config = OmegaConf.create()
        for dataset_name in datasets:
            dataset_config_path = 'apps/language_models/src/pipelines/minigpt4_utils/configs/cc_sbu_align.yaml'
            dataset_config = OmegaConf.merge(
                dataset_config,
                OmegaConf.load(dataset_config_path),
                {"datasets": {dataset_name: config["datasets"][dataset_name]}},
            )
        dataset_config = dataset_config['datasets']
        vis_processor_cfg = dataset_config.cc_sbu_align.vis_processor.train
        vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)
        print("Initialization complete")

        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to('cpu'),
                          torch.tensor([2277, 29937]).to('cpu')]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        self.first_llama = None
        self.second_llama = None
    
    # Currently we're compiling VisionModel for fp32/cuda.
    def compile_vision_model(self):
        # vmfb = get_vmfb_from_path(
        #     Path("minigpt4_vision_model_fp16_cuda.vmfb"), self.device, "tm_tensor"
        # )
        # return vmfb
        if not self._compile:
            vmfb = get_vmfb_from_path(
                self.vision_model_vmfb_path, self.device, "tm_tensor"
            )
            if vmfb is not None:
                return vmfb

        visionModel = VisionModel(self.model.ln_vision, self.model.visual_encoder)
        extended_model_name = f"minigpt4_vision_model_{self.precision}_{self.device}"
        print(f"Going to compile {extended_model_name}")
        # Inputs for VisionModel.
        inputs = [torch.randint(3, (1, 3, 224, 224), dtype=torch.float32)]
        is_f16 = False
        if self.precision == "fp16":
            is_f16 = True
        shark_visionModel, _ = compile_through_fx(
            visionModel,
            inputs,
            extended_model_name=extended_model_name,
            is_f16=is_f16,
            debug=False,
            generate_vmfb=True,
            save_dir=os.getcwd(),
            extra_args=[],
            base_model_id=None,
            model_name=extended_model_name,
            precision=None,
            return_mlir=False,
            device=self.device,
        )
        print(f"Generated {extended_model_name}.vmfb")
        return shark_visionModel
    
    def compile_qformer_model(self):
        if not self._compile:
            vmfb = get_vmfb_from_path(
                self.qformer_bert_model_vmfb_path, self.device, "tm_tensor"
            )
            if vmfb is not None:
                return vmfb
        
        qformerBertModel = QformerBertModel(self.model.Qformer.bert)
        extended_model_name = f"minigpt4_qformer_bert_model_{self.precision}_{self.device}"
        print(f"Going to compile {extended_model_name}")
        # Inputs for QFormer.
        inputs = [torch.randint(3, (1, 32, 768), dtype=torch.float32),
                  torch.randint(3, (1, 257, 1408), dtype=torch.float32),
                  torch.randint(3, (1, 257), dtype=torch.int64)]
        is_f16 = False
        f16_input_mask = []
        if self.precision == "fp16":
            is_f16 = True
            f16_input_mask = [True, True, False]
        shark_QformerBertModel, _ = compile_through_fx(
            qformerBertModel,
            inputs,
            extended_model_name=extended_model_name,
            is_f16=is_f16,
            f16_input_mask=f16_input_mask,
            debug=False,
            generate_vmfb=True,
            save_dir=os.getcwd(),
            extra_args=[],
            base_model_id=None,
            model_name=extended_model_name,
            precision=None,
            return_mlir=False,
            device=self.device,
        )
        print(f"Generated {extended_model_name}.vmfb")
        return shark_QformerBertModel
    
    def compile_first_llama(self):
        if not self._compile:
            vmfb = get_vmfb_from_path(
                self.first_llama_vmfb_path, self.device, "tm_tensor"
            )
            if vmfb is not None:
                self.first_llama = vmfb
                return vmfb
        
        firstLlamaModel = FirstLlamaModel(self.model.llama_model)
        extended_model_name = f"minigpt4_first_llama_{self.precision}_{self.device}"
        print(f"Going to compile {extended_model_name}")
        # Inputs for FirstLlama.
        inputs_embeds = torch.ones((1,170, 4096), dtype=torch.float32)
        position_ids = torch.ones((1,170), dtype=torch.int64)
        attention_mask = torch.ones((1,170), dtype=torch.int32)
        inputs = [inputs_embeds, position_ids, attention_mask]
        is_f16 = False
        f16_input_mask = []
        if self.precision == "fp16":
            is_f16 = True
            f16_input_mask= [True, False, False]
        shark_firstLlamaModel, _ = compile_through_fx(
            firstLlamaModel,
            inputs,
            extended_model_name=extended_model_name,
            is_f16=is_f16,
            f16_input_mask=f16_input_mask,
            debug=False,
            generate_vmfb=True,
            save_dir=os.getcwd(),
            extra_args=[],
            base_model_id=None,
            model_name=extended_model_name,
            precision=None,
            return_mlir=False,
            device=self.device,
        )
        print(f"Generated {extended_model_name}.vmfb")
        self.first_llama = shark_firstLlamaModel
        return shark_firstLlamaModel
    
    def compile_second_llama(self):
        if not self._compile:
            vmfb = get_vmfb_from_path(
                self.second_llama_vmfb_path, self.device, "tm_tensor"
            )
            if vmfb is not None:
                self.second_llama = vmfb
                return vmfb
        
        secondLlamaModel = SecondLlamaModel(self.model.llama_model)
        extended_model_name = f"minigpt4_second_llama_{self.precision}_{self.device}"
        print(f"Going to compile {extended_model_name}")
        # Inputs for SecondLlama.
        input_ids = torch.zeros((1,1), dtype=torch.int64)
        position_ids = torch.zeros((1,1), dtype=torch.int64)
        attention_mask = torch.zeros((1,171), dtype=torch.int32)
        past_key_value = []
        for i in range(64):
            past_key_value.append(torch.zeros(1,32,170,128, dtype=torch.float32))
        inputs = [input_ids, position_ids, attention_mask, *past_key_value]
        is_f16 = False
        f16_input_mask = []
        if self.precision == "fp16":
            is_f16 = True
            f16_input_mask = [False, False, False]
            for i in past_key_value:
                f16_input_mask.append(True)

        shark_firstLlamaModel, _ = compile_through_fx(
            secondLlamaModel,
            inputs,
            extended_model_name=extended_model_name,
            is_f16=is_f16,
            f16_input_mask=f16_input_mask,
            debug=False,
            generate_vmfb=True,
            save_dir=os.getcwd(),
            extra_args=[],
            base_model_id=None,
            model_name=extended_model_name,
            precision=None,
            return_mlir=False,
            device=self.device,
        )
        print(f"Generated {extended_model_name}.vmfb")
        self.second_llama = shark_firstLlamaModel
        return shark_firstLlamaModel
        
    # Not yet sure why to use this.
    def compile(self):
        pass

    # Going to use `answer` instead.
    def generate(self, prompt):
        pass

    # Might use within `answer`, if needed.
    def generate_new_token(self, params):
        pass

    # Not needed yet because MiniGPT4BaseModel already loads this - will revisit later,
    # if required.
    def get_tokenizer(self):
        pass

    # DumDum func - doing the intended stuff already at MiniGPT4BaseModel,
    # i.e load llama, etc.
    def get_src_model(self):
        pass

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list, max_length - max_new_tokens)

        current_max_len = embs.shape[1] + max_new_tokens

        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        #########################################################################################################
        
        generation_config = GenerationConfig.from_model_config(self.model.llama_model.config)
        kwargs = {
            "inputs_embeds":embs,
            "max_new_tokens":max_new_tokens,
            "num_beams":num_beams,
            "do_sample":True,
            "min_length":min_length,
            "top_p":top_p,
            "repetition_penalty":repetition_penalty,
            "length_penalty":length_penalty,
            "temperature":temperature,
        }
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        logits_processor = LogitsProcessorList()
        stopping_criteria = self.stopping_criteria
        inputs = None
        inputs_tensor, model_input_name, model_kwargs = self.model.llama_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        generation_config.pad_token_id = self.model.llama_tokenizer.pad_token_id
        pad_token_id = generation_config.pad_token_id
        embs_for_pad_token_id = self.model.llama_model.model.embed_tokens(torch.tensor([pad_token_id]))
        model_kwargs["attention_mask"] = torch.logical_not(
                torch.tensor([torch.all(torch.eq(inputs_tensor[:,d,:], embs_for_pad_token_id)).int() for d in range(inputs_tensor.shape[1])]
                            ).unsqueeze(0)
        ).int()
        attention_meta_data = (model_kwargs["attention_mask"][0] == 0).nonzero(as_tuple=True)[0]
        first_zero = attention_meta_data[0].item()
        last_zero = attention_meta_data[-1].item()
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        input_ids_seq_length = input_ids.shape[-1]
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        logits_warper = self.model.llama_model._get_logits_warper(generation_config)
        input_ids, model_kwargs = self.model.llama_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=False,
            **model_kwargs,
        )
        # DOUBT: stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        scores = None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        i = 0
        timesRan = 0
        is_fp16 = True
        llama_list = []
        isPyTorchVariant = False
        while True:
            print("****** Iteration %d ******" % (i))
            # prepare model inputs
            model_inputs = self.model.llama_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            if i == 0:
                shark_inputs = []
                if is_fp16:
                    model_inputs['inputs_embeds'] = model_inputs['inputs_embeds'].to(torch.float16)
                shark_inputs.append(model_inputs['inputs_embeds'].detach())
                shark_inputs.append(model_inputs['position_ids'].detach())
                shark_inputs.append(model_inputs['attention_mask'].detach())

                if self.first_llama is None:
                    self.compile_first_llama()
                outputs_shark = self.first_llama("forward", shark_inputs)
                outputs = []
                for out_shark in outputs_shark:
                    outputs.append(torch.from_numpy(out_shark))
                del outputs_shark
            else:
                shark_inputs = []
                shark_inputs.append(model_inputs['input_ids'].detach())
                shark_inputs.append(model_inputs['position_ids'].detach())
                shark_inputs.append(model_inputs['attention_mask'].detach())
                for pkv in list(model_inputs['past_key_values']):
                    shark_inputs.append(pkv.detach())
                if self.second_llama is None:
                    self.compile_second_llama()
                outputs_shark = self.second_llama("forward", shark_inputs)
                outputs = []
                for out_shark in outputs_shark:
                    outputs.append(torch.from_numpy(out_shark))
                del outputs_shark

            outputs_logits = outputs[0]
            next_token_logits = outputs_logits[:, -1, :]
            if is_fp16:
                next_token_logits = next_token_logits.to(torch.float32)
            
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            outputs_for_update_func = {
            }
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model.llama_model._update_model_kwargs_for_generation(
                outputs_for_update_func, model_kwargs, is_encoder_decoder=False
            )
            model_kwargs['past_key_values'] = outputs[1:]
            print(type(model_kwargs['past_key_values']))
            if timesRan >=1 :
                tmp_attention_mask = torch.cat((model_kwargs['attention_mask'][:, :first_zero],model_kwargs['attention_mask'][:, first_zero+1:]), dim=1)
                model_kwargs['attention_mask'] = tmp_attention_mask
                pkv_list = []
                for pkv_pair_tuple in model_kwargs['past_key_values']:
                    x = torch.cat((pkv_pair_tuple[:,:,:first_zero,:], pkv_pair_tuple[:,:,first_zero+1:,:]), dim=2)
                    if is_fp16:
                        x = x.to(torch.float16)
                    pkv_list.append(x)
                model_kwargs['past_key_values'] = tuple(pkv_list)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            
            i = i + 1
            timesRan += 1
            output_token = input_ids[0] 


            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            
            print("Out: ", str(output_text))
        llama_list.clear()
        output_token = input_ids[0]


        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to('cpu')
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to('cpu')
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to('cpu')

        device = image.device
        if self.model.low_resource:
            self.model.vit_to_cpu()
            image = image.to("cpu")

        with self.model.maybe_autocast():
            shark_visionModel = self.compile_vision_model()
            if self.precision == "fp16":
                image = image.to(torch.float16)
            image_embeds = shark_visionModel("forward", (image,))
            # image_embeds = shark_visionModel.forward(image)
            image_embeds = torch.from_numpy(image_embeds)
            image_embeds = image_embeds.to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
            if self.precision == "fp16":
                query_tokens = query_tokens.to(torch.float16)
            shark_QformerBertModel = self.compile_qformer_model()
            query_output = shark_QformerBertModel("forward", (query_tokens, image_embeds, image_atts,))
            # query_output = shark_QformerBertModel.forward(query_tokens, image_embeds, image_atts)
            query_output = torch.from_numpy(query_output)

            inputs_llama = self.model.llama_proj(query_output)
        image_emb = inputs_llama
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        return msg

    #"""
    def get_context_emb(self, conv, img_list, max_allowed_tokens=200):
        self.model.llama_tokenizer.padding_side = "left"
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        prompt_segs_pre = prompt_segs[:-1]
        seg_tokens_pre = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to('cpu').input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs_pre)
        ]
        self.model.llama_model.config.pad_token_id = self.model.llama_tokenizer.pad_token_id
        
        seg_embs_pre = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_pre]
        mixed_embs_pre = [emb.to('cpu') for pair in zip(seg_embs_pre, img_list) for emb in pair]
        mixed_embs_pre = torch.cat(mixed_embs_pre, dim=1)
        max_allowed_tokens = max_allowed_tokens - mixed_embs_pre.shape[1]
        final_prompt = prompt_segs[-1]
        seg_tokens_post = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", padding="max_length", max_length=max_allowed_tokens, add_special_tokens=False).to('cpu').input_ids
            # only add bos to the first seg
            for i, seg in enumerate([final_prompt])
        ]
        seg_tokens_post = seg_tokens_post[0]
        seg_embs_post = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_post]
        mixed_embs_post = [seg_embs_post[0].to('cpu')]
        mixed_embs_post = torch.unsqueeze(mixed_embs_post[0], 0)
        mixed_embs = [mixed_embs_pre] + [mixed_embs_post]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
        
if __name__ == "__main__":
    args = parser.parse_args()

    device = args.device
    precision = args.precision
    _compile = args.compile

    if args.image_path == "":
        print("To run MiniGPT4 in CLI mode please provide an image's path using --image_path")
        sys.exit()

    vision_model_mlir_path = (
        Path("minigpt4_vision_model_fp16.mlir")
        if args.vision_model_mlir_path is None
        else Path(args.vision_model_mlir_path)
    )
    vision_model_vmfb_path = (
        Path("minigpt4_vision_model_fp16_cuda.vmfb")
        if args.vision_model_vmfb_path is None
        else Path(args.vision_model_vmfb_path)
    )
    qformer_bert_model_mlir_path = (
        Path("minigpt4_qformer_bert_model_fp32.mlir")
        if args.qformer_bert_model_mlir_path is None
        else Path(args.qformer_bert_model_mlir_path)
    )
    qformer_bert_model_vmfb_path = (
        Path("minigpt4_qformer_bert_model_fp32_cuda.vmfb")
        if args.qformer_bert_model_vmfb_path is None
        else Path(args.qformer_bert_model_vmfb_path)
    )
    first_llama_mlir_path = (
        Path("minigpt4_first_llama_fp16.mlir")
        if args.first_llama_mlir_path is None
        else Path(args.first_llama_mlir_path)
    )
    first_llama_vmfb_path = (
        Path("minigpt4_first_llama_fp16_cuda.vmfb")
        if args.first_llama_vmfb_path is None
        else Path(args.first_llama_vmfb_path)
    )
    second_llama_mlir_path = (
        Path("minigpt4_second_llama_fp16.mlir")
        if args.second_llama_mlir_path is None
        else Path(args.second_llama_mlir_path)
    )
    second_llama_vmfb_path = (
        Path("minigpt4_second_llama_fp16_cuda.vmfb")
        if args.second_llama_vmfb_path is None
        else Path(args.second_llama_vmfb_path)
    )
    chat = MiniGPT4(model_name="MiniGPT4", hf_model_path=None, max_new_tokens=30,
                    device=device, precision=precision, _compile=_compile,
                    vision_model_mlir_path=vision_model_mlir_path,
                    vision_model_vmfb_path=vision_model_vmfb_path,
                    qformer_bert_model_mlir_path=qformer_bert_model_mlir_path,
                    qformer_bert_model_vmfb_path=qformer_bert_model_vmfb_path,
                    first_llama_mlir_path=first_llama_mlir_path,
                    first_llama_vmfb_path=first_llama_vmfb_path,
                    second_llama_mlir_path=second_llama_mlir_path,
                    second_llama_vmfb_path=second_llama_vmfb_path
                )

    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(args.image_path, chat_state, img_list)
    print("Uploaded image successfully to the bot. You may now start chatting with the bot. Enter 'END' without quotes to end the interaction")
    continue_execution = True

    while continue_execution:
        user_message = input("User: ")
        if user_message == "END":
            print("Bot: Good bye.\n")
            break
        chat.ask(user_message, chat_state)
        bot_message = chat.answer(conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1.0,
                    max_new_tokens=30,
                    max_length=200)[0]
        print("Bot: ",bot_message)