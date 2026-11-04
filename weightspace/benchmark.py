"""Synthetic local benchmark: reports time/RSS; does not create files."""
from __future__ import annotations
import argparse
import platform
import resource
import time
import torch
from .metrics import spectral,basis,VectorAccumulator


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--side',type=int,default=512)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    rng=torch.Generator(device='cpu').manual_seed(42)
    x=torch.randn(a.side,a.side,generator=rng)
    y=x+torch.randn(a.side,a.side,generator=rng)*.01
    measurements=[]
    for label,fn in [('vector',lambda: VectorAccumulator().update(x.double(),y.double())),
                     ('full_svd',lambda: spectral(x,'exact',a.side*a.side,32,a.device)),
                     ('principal_basis',lambda:basis(x,8,a.device,a.side*a.side))]:
        t=time.perf_counter();fn()
        if a.device=='cuda':torch.cuda.synchronize()
        measurements.append((label,round(time.perf_counter()-t,3)))
    print(dict(platform=platform.platform(),torch=torch.__version__,size=(a.side,a.side),
               device=a.device,seconds=dict(measurements),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               note='synthetic matrix microbenchmark; not a production completion-time estimate'))
if __name__=='__main__':main()
