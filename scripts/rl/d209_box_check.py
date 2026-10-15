"""Compare before/after written boxes on the same cached held-out frames."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from peft import PeftModel
import train_uav_flow_vla as T
from box_eval import evaluate_boxes
from uav_flow_action_tokenizer import ActionTokenizer
p=argparse.ArgumentParser()
p.add_argument("--root",type=Path,required=True)
p.add_argument("--adapter",required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
rows=json.loads((a.root / "box_check/rows.json").read_text())
api, processor, base, _=T.setup("qwen","bf16")
model=PeftModel.from_pretrained(base,a.adapter)
model.eval()
tokenizer=ActionTokenizer(processor.tokenizer,T.official_stats())
def image_fn(row):
    im=Image.open(a.root / "box_check" / row["file"]).convert("RGB")
    im.thumbnail((256,256))
    return im
result=evaluate_boxes(model,processor,tokenizer,rows,8,api.cuda,image_fn=image_fn)
a.out.write_text(json.dumps(dict(adapter=a.adapter,**result),indent=2))
print(json.dumps(result))
