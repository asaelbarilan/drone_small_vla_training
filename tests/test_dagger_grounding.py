"""CPU tests of production encoding functions without loading a VLM."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
def functions(path, names, env):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)

class GroundingTest(unittest.TestCase):
    def setUp(self):
        self.env = {"torch": torch, "ACTION_HEAD": {"on": False}}
        functions(ROOT / "scripts/train_uav_flow_vla.py",
                  {"encode", "append_ids", "box_ids", "progress_ids", "box_text", "next_box_text", "frame_key"}, self.env)
        functions(ROOT / "scripts/rl/dagger_relabel.py", {"box_context"}, self.env)
        class Processor:
            tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()), eos_token_id=999)
            def apply_chat_template(self, *a, **kw): return "prompt"
            def __call__(self, **kw):
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        self.processor = Processor()
        self.tokenizer = SimpleNamespace(token_ids=lambda chunk: [801, 802])
        self.row = {"prompt": "x", "progress_text": "Left +01.0", "chunk": [[1, 0, 0, 0]]}
    def encoded(self, row, answer=True):
        return self.env["encode"](self.processor, self.tokenizer, row, 1, image=object(), with_answer=answer)
    def test_student_prefix_mask_and_action_supervision(self):
        ctx = self.env["box_context"]({"box": {"now": [10,20,30,40], "next": []}})
        row = {**self.row, **ctx}
        batch = self.encoded(row)
        prefix = self.env["box_ids"](self.processor, row)
        self.assertEqual(batch["input_ids"][0,3:3+len(prefix)].tolist(), prefix)
        self.assertTrue((batch["labels"][0,:3+len(prefix)] == -100).all())
        self.assertEqual(batch["labels"][0,-3:].tolist(), [801,802,999])
        self.assertNotEqual(batch["labels"][0,3+len(prefix)].item(), -100)
    def test_independent_boxes_remain_supervised(self):
        row = {**self.row, "box": [10,20,30,40], "grounding_only": True}
        self.assertNotEqual(self.encoded(row)["labels"][0,3].item(), -100)
    def test_null_current_and_absent_next_match_server(self):
        ctx = self.env["box_context"]({"box": {"now": None, "next": None}})
        self.assertEqual(ctx, {"box_context": True, "box": []})
        with self.assertRaises(ValueError): self.env["box_context"]({})
    def test_context_cannot_be_grounding_target(self):
        with self.assertRaises(ValueError): self.encoded({**self.row, "box": [], "box_context": True, "grounding_only": True})
    def test_generation_context_order(self):
        row = {**self.row, "box": [], "box_context": True}
        expected = [1,2,3] + self.env["box_ids"](self.processor,row) + list(row["progress_text"].encode())
        self.assertEqual(self.encoded(row,False)["input_ids"][0].tolist(),expected)
    def test_portable_frame_identity(self):
        self.assertEqual(self.env["frame_key"]("C:\\store\\frames\\sim_x\\00001.jpg"), "sim_x/00001.jpg")
        self.assertEqual(self.env["frame_key"]("/home/ubuntu/frames/sim_x/00001.jpg"), "sim_x/00001.jpg")

if __name__ == "__main__": unittest.main()
