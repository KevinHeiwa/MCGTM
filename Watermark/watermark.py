
import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
from collections import OrderedDict
import numpy # for gradio hot reload
import gradio as gr
from tqdm import tqdm
import torch
import json
import logging
from watermark_global import *
import time

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


os.environ["CUDA_VISIBLE_DEVICES"] = " "
model_path = " "

def read_file(filename):
    json_objs = []
    with open(filename, "r") as file:
        for line in file:
            json_obj = json.loads(line, strict=False)
            json_objs.append(json_obj)
    return json_objs

def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n"+ "\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_path,
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,       
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
    )
    # modify 0.25-->0.5 test
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=" ",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map='auto',trust_remote_code=True,
        mirror='tuna').to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    vocab_dict = tokenizer.get_vocab()
    ordered_vocab_dict = OrderedDict(sorted(vocab_dict.items(), key=lambda x: x[1]))
    vocab = list(ordered_vocab_dict.values())
    watermark_processor = WatermarkLogitsProcessor(vocab,
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens,
                                                    tokenizer=tokenizer)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)
    output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
    output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]
    
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    
    vocab_dict = tokenizer.get_vocab()
    ordered_vocab_dict = OrderedDict(sorted(vocab_dict.items(), key=lambda x: x[1]))
    vocab = list(ordered_vocab_dict.values())
    watermark_detector = WatermarkDetector(vocab,
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    
  
    return output, args


def convert_to_binary(str1, str2):
    binary_str1 = ' '.join(format(int(char), '04b') for char in str1)
    binary_str2 = ' '.join(format(int(char), '04b') for char in str2)
    combined_binary = binary_str1 + ' 1111 ' + binary_str2
    return combined_binary


def main(args): 
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    prompt_data = read_file(args.prompt_file)
    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    if not args.skip_model_load:
        with open(args.output_file,'w') as outfile:
            # Generate and detect, report to stdout
            for idx, cur_prompt in tqdm(enumerate(prompt_data)):

                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                # 重制round_time
                reset_water_round()
                sys.stdout = Logger(f"/system_{idx}.json", sys.stdout)
                sys.stderr = Logger(f"/system_{idx}.log", sys.stderr)
                
                input_text = cur_prompt['text']
                tmp=tokenizer(input_text,return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.max_new_tokens).to(device)   #
                args.default_prompt = input_text

                term_width = 80
                _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                                    args, 
                                                                                                    model=model, 
                                                                                                    device=device, 
                                                                                                    tokenizer=tokenizer)
                without_watermark_detection_result= detect(decoded_output_without_watermark, 
                                                            args, 
                                                            device=device, 
                                                            tokenizer=tokenizer)
                with_watermark_detection_result= detect(decoded_output_with_watermark, 
                                                        args, 
                                                        device=device, 
                                                        tokenizer=tokenizer)
            
                print("#"*term_width)
                print("Output without watermark:")
                print(decoded_output_without_watermark)
                print("-"*term_width)
                print("#"*term_width)
                print("Output with watermark:")
                print(decoded_output_with_watermark)
                print("-"*term_width)
                pprint(with_watermark_detection_result)
                try:#string too short error
                    tmp={'idx':idx,'decoded_output_without_watermark':decoded_output_without_watermark,'decoded_output_with_watermark':decoded_output_with_watermark,
                         'ICML_result':str(with_watermark_detection_result)}
                            
                except:
                    pass    
                sys.stdout.flush()
                sys.stderr.flush()
                json.dump(tmp,outfile)
                outfile.write('\n')
        outfile.close()
    return

if __name__ == "__main__":
    time_start = time.time()
    args = parse_args()
    main(args)
    time_end = time.time()
    time_sum = time_start -time_end