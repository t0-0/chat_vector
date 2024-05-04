import argparse

import torch  # torch_dtypeで必要
import torch.nn.functional as F
from huggingface_hub import create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--base_model", type=str, required=True)
parser.add_argument("-cm", "--chat_model", type=str, required=True)
parser.add_argument("-om", "--original_model", type=str, required=True)
parser.add_argument("-dt", "--dtype", type=str, default="torch.bfloat16")
parser.add_argument(
    "-ufa", "--use_flash_attention2", action="store_true", default=False
)
parser.add_argument("-sl", "--skip_layernorm", action="store_true", default=False)
parser.add_argument(
    "-sel", "-skip_embedding_lmhead", action="store_true", default=False
)
parser.add_argument("-c", "--coefficient", type=float, default=1)
parser.add_argument("-debug", "--debug", action="store_true", default=False)
args = parser.parse_args()

if not "torch." in args.dtype:
    exit(1)

chat_tokenizer = AutoTokenizer.from_pretrained(args.chat_model, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    args.original_model,
    trust_remote_code=True,
    chat_template=chat_tokenizer.chat_template,
)

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=eval(args.dtype),
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attention2 else None,
)
chat_model = AutoModelForCausalLM.from_pretrained(
    args.chat_model,
    torch_dtype=eval(args.dtype),
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attention2 else None,
)

original_model = AutoModelForCausalLM.from_pretrained(
    args.original_model,
    torch_dtype=eval(args.dtype),
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attention2 else None,
)

skip_parameters = ["model.embed_tokens.weight", "lm_head.weight"]

for k, v in original_model.state_dict().items():
    if args.skip_embedding_lmhead and k in skip_parameters:
        continue
    if args.skip_layernorm and "layernorm" in k:
        continue
    chat_vector = chat_model.state_dict()[k] - base_model.state_dict()[k]
    if k in skip_parameters:
        chat_vector = F.pad(chat_vector, (0, 0, 0, v.size()[0] - chat_vector.size()[0]))
    v.copy_(v + chat_vector * args.coefficient)

if args.debug:
    original_model.save_pretrained("new_model")
    tokenizer.save_pretrained("new_model")

repo_name = f"{args.original_model}_add-chat-vector"
if not args.skip_embedding_lmhead:
    repo_name += "-including-EM-LM"
if args.skip_layernorm:
    repo_name += "-except-LN"
if args.coefficient != 1:
    repo_name += f"-{args.coefficient}"
try:
    create_repo(repo_name, repo_type="model", private=True)
except Exception as e:
    print(f"repo {repo_name} already exists!")
    exit(1)
original_model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
