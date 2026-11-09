"""Strict checkpoint loading and synthetic CPU forward; never opens a dataset."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('PEMS03','PEMS04','PEMS07','PEMS08','PEMSBAY'))
    args = parser.parse_args()
    import torch
    from models.Scope import Model
    torch.set_num_threads(2)
    torch.manual_seed(0)
    runs = json.loads((ROOT/'tools/final_runs.json').read_text())
    manifest = json.loads((ROOT/'tools/final_assets_manifest.json').read_text())
    hashes = {x['path']: x['sha256'] for x in manifest['files']}
    nodes = dict(PEMS03=358, PEMS04=307, PEMS07=883, PEMS08=170, PEMSBAY=325)
    results=[]
    for ds in args.datasets or list(runs):
        try:
            base=ROOT/runs[ds]
            config=json.loads((base/'config.json').read_text(encoding='utf-8-sig'))
            if 'model_config' in config:
                raise ValueError('Expected saved flat runtime config, not a training template')
            config.update(device='cpu',num_nodes=nodes[ds],pearson_adj=None)
            config.setdefault('prob_factor',5)
            if isinstance(config.get('k_hops'),str):
                config['k_hops']=[int(x) for x in config['k_hops'].split(',')]
            checkpoint=base/'best.pth'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest!=hashes[runs[ds]+'/best.pth']:
                raise ValueError('Checkpoint differs from verified local manifest')
            model=Model(SimpleNamespace(**config)).cpu().eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
            t,h,n=config['input_length'],config['predict_length'],nodes[ds]
            x=torch.randn(1,t,n,config['in_dim'])
            xt=torch.zeros(1,t,2,dtype=torch.long)
            yt=torch.zeros(1,h,2,dtype=torch.long)
            reference=torch.randn(1,h,n,config['pre_dim'])
            available=torch.ones_like(reference)
            with torch.inference_mode():
                result=model([x,xt,yt,reference,reference,available])
            prediction=result[0] if isinstance(result,(tuple,list)) else result
            if tuple(prediction.shape)!=(1,h,n,config['pre_dim']) or not torch.isfinite(prediction).all():
                raise ValueError('Invalid synthetic output shape or nonfinite values')
            results.append(dict(dataset=ds,status='PASS',shape=list(prediction.shape),
                                parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest))
        except Exception as exc:
            results.append(dict(dataset=ds,status='FAIL',error=str(exc)))
    print(json.dumps({'scope':'synthetic only; no data, training, calibration or Test evaluation', 'results':results},indent=2))
    return int(any(x['status']!='PASS' for x in results))


if __name__=='__main__':sys.exit(main())
