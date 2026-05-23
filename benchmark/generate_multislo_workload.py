#!/usr/bin/env python3
"""
Synthetic deterministic workload generator used by unit tests.

NOTE:
- This file is intentionally lightweight and synthetic.
- Do NOT use it to replace benchmark/gen_multi_slo_workload.py in real
  experiments; the production benchmark pipeline should continue to use
  benchmark/gen_multi_slo_workload.py and existing workload JSONL inputs.
"""
from __future__ import annotations
import argparse, json, random, math
from typing import List

SLO_CLASSES={"tight":30.0,"medium":60.0,"loose":120.0}

def gen_arrivals(n:int, mode:str, interval:float, rate:float, rng:random.Random)->List[float]:
    t=0.0; out=[]
    for _ in range(n):
        if mode=="fixed":
            out.append(t); t += interval
        else:
            t += rng.expovariate(rate); out.append(t)
    return out

def sample_len(spec:str, rng:random.Random)->int:
    kind,*rest=spec.split(":")
    if kind=="fixed": return int(rest[0])
    if kind=="uniform":
        a,b=map(int,rest); return rng.randint(a,b)
    if kind=="normal":
        mu,s=map(float,rest); return max(1,int(round(rng.gauss(mu,s))))
    raise ValueError("bad dist")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--num-requests',type=int,required=True)
    p.add_argument('--prompt-len-dist',default='uniform:16:128')
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--arrival-process',choices=['fixed','poisson'],default='fixed')
    p.add_argument('--fixed-interval-sec',type=float,default=0.1)
    p.add_argument('--poisson-rate',type=float,default=5.0)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    rng=random.Random(args.seed)
    arr=gen_arrivals(args.num_requests,args.arrival_process,args.fixed_interval_sec,args.poisson_rate,rng)
    classes=["tight","medium","loose"]
    with open(args.out,'w',encoding='utf-8') as f:
        for i in range(args.num_requests):
            n=sample_len(args.prompt_len_dist,rng)
            input_ids=[rng.randint(10,20000) for _ in range(n)]
            cls=classes[i%3]
            row={
                "request_id":f"req-{i:06d}","input_ids":input_ids,
                "arrival_offset_sec":arr[i],"slo_tpot_ms":SLO_CLASSES[cls],"slo_class":cls,
                "max_tokens":args.max_tokens,"temperature":0.0,"ignore_eos":False,
                "category":"synthetic"
            }
            f.write(json.dumps(row,ensure_ascii=False)+"\n")

if __name__=='__main__': main()
