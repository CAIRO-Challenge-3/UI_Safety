"""
Export an Ultralytics YOLO .pt detector into a form Pepper's OpenCV 4.1.2 can run.

Run on your laptop (Python 3 with ultralytics + onnx installed):
    python export_pepper_onnx.py --weights models/helmet-26s.pt --out models/helmet_pepper

Writes a folder (copy the whole folder to Pepper):
    seg0.onnx, seg1.onnx, ...   graph pieces OpenCV runs
    manifest.json               how to chain them, class names, head layout

OpenCV 4.1.2 can't import a stock Ultralytics export, so this:
  1. exports at opset 10 (no empty "roi" tensor in Resize),
  2. turns Resize into Upsample with a "scales" attribute (the only form 4.1.2 sizes correctly),
  3. cuts the graph before the Detect head's reshape/decoding; ppe_pepper_onboard.py
     decodes the raw per-stride box/class maps in numpy,
  4. cuts out attention blocks (C2PSA in YOLO11/26: MatMul between two activations,
     which 4.1.2 can't do) and records them as a few numpy ops instead. The rest
     of the graph is split into segments around them.
YOLOv8 models have no attention and come out as a single segment.
"""

import argparse
import json
import os
import shutil
import tempfile

import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.utils import extract_model
from ultralytics import YOLO

# Ops inside an attention block that run in numpy on the robot.
NUMPY_OPS = {"Split", "Mul", "Transpose", "MatMul", "Softmax", "Reshape"}


def export_opset10(weights, imgsz, workdir):
    pt = os.path.join(workdir, "model.pt")
    shutil.copy(weights, pt)
    yolo = YOLO(pt)
    head_index = len(yolo.model.model) - 1     # Detect is the last layer
    names = [yolo.names[i] for i in range(len(yolo.names))]
    return yolo.export(format="onnx", imgsz=imgsz, opset=10), head_index, names


def attention_nodes(graph):
    """Nodes of every attention block from its first Split to the Reshape after the last MatMul."""
    prefixes = {n.name.rsplit("/", 1)[0] + "/" for n in graph.node
                if n.op_type == "MatMul" and "/attn/" in n.name}
    core = set()
    for prefix in prefixes:
        block = [i for i, n in enumerate(graph.node) if n.name.startswith(prefix)]
        split = next(i for i in block if graph.node[i].op_type == "Split")
        last_matmul = max(i for i in block if graph.node[i].op_type == "MatMul")
        tail = next(i for i in block if i > last_matmul and graph.node[i].op_type == "Reshape")
        core.update(i for i in block if split <= i <= tail and graph.node[i].op_type in NUMPY_OPS)
    return core


def numpy_op(node, initializers):
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    consts = {name: numpy_helper.to_array(initializers[name]).tolist()
              for name in node.input if name in initializers}
    return {"op": node.op_type, "inputs": list(node.input), "outputs": list(node.output),
            "attrs": attrs, "consts": consts}


def split_graph(model, head_index):
    """Assign nodes to OpenCV segments / numpy blocks; return the plan."""
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    core = attention_nodes(graph)
    prefix = "/model.%d/" % head_index
    head_outputs = [n.input[0] for n in graph.node
                    if n.op_type == "Reshape" and n.name.startswith(prefix) and "dfl" not in n.name]

    # Segment level of each tensor: a numpy block bumps the level of what it produces.
    level, producer, node_level = {graph.input[0].name: 0}, {}, {}
    for i, node in enumerate(graph.node):
        ins = [t for t in node.input if t and t not in initializers]
        if i in core:
            lvl = max([level[t] + (0 if producer.get(t) in core else 1) for t in ins] or [0])
        else:
            lvl = max([level[t] for t in ins] or [0])
        node_level[i] = lvl
        for t in node.output:
            level[t], producer[t] = lvl, i
    used = set()
    stack = list(head_outputs)
    while stack:                              # only nodes the head outputs depend on
        t = stack.pop()
        i = producer.get(t)
        if i is None or i in used:
            continue
        used.add(i)
        stack.extend(x for x in graph.node[i].input if x and x not in initializers)

    def group(i):
        return ("np", node_level[i]) if i in core else ("cv", node_level[i])

    consumers = {}
    for i in used:
        for t in graph.node[i].input:
            consumers.setdefault(t, set()).add(i)

    segments, blocks = {}, {}
    for i in sorted(used):
        key = group(i)
        target = blocks if key[0] == "np" else segments
        target.setdefault(key[1], []).append(i)

    # A numpy block at level L consumes level L-1 tensors and feeds segment L.
    plan = []
    for lvl in sorted(set(segments) | set(blocks)):
        if lvl in blocks:
            plan.append({"kind": "numpy",
                         "ops": [numpy_op(graph.node[i], initializers) for i in sorted(blocks[lvl])]})
        if lvl in segments:
            members = set(segments[lvl])
            inputs, outputs = [], []
            for i in sorted(members):
                for t in graph.node[i].input:
                    if t and t not in initializers and producer.get(t) not in members and t not in inputs:
                        inputs.append(t)
                for t in graph.node[i].output:
                    outside = consumers.get(t, set()) - members
                    if (outside or t in head_outputs) and t not in outputs:
                        outputs.append(t)
            plan.append({"kind": "opencv", "inputs": inputs, "outputs": outputs})
    return plan, head_outputs


def write_segment(full_path, out_path, inputs, outputs):
    extract_model(full_path, out_path, inputs, outputs)
    seg = onnx.load(out_path)
    # OpenCV forwards by layer name = a node's first output; expose other outputs via Identity.
    first_outputs = {n.output[0] for n in seg.graph.node}
    renamed = []
    for vi in seg.graph.output:
        if vi.name not in first_outputs:
            alias = vi.name + "__out"
            seg.graph.node.append(helper.make_node("Identity", [vi.name], [alias]))
            renamed.append((vi, alias))
    for vi, alias in renamed:
        vi.name = alias
    # Resize -> Upsample(scales attribute) for OpenCV 4.1.2.
    values = {t.name: numpy_helper.to_array(t) for t in seg.graph.initializer}
    values.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t)
                   for n in seg.graph.node if n.op_type == "Constant"})
    for node in seg.graph.node:
        if node.op_type == "Resize":
            scales = values[node.input[-1]]
            node.op_type = "Upsample"
            del node.input[1:]
            node.attribute.append(helper.make_attribute("scales", [float(s) for s in scales]))
    onnx.save(seg, out_path)
    return [alias for _, alias in renamed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Ultralytics YOLO detect .pt model")
    parser.add_argument("--out", required=True, help="output folder for Pepper")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    workdir = tempfile.mkdtemp()
    try:
        exported, head_index, names = export_opset10(args.weights, args.imgsz, workdir)
        model = shape_inference.infer_shapes(onnx.load(exported))
        full = os.path.join(workdir, "inferred.onnx")
        onnx.save(model, full)
        plan, head_outputs = split_graph(model, head_index)

        if os.path.isdir(args.out):
            shutil.rmtree(args.out)
        os.makedirs(args.out)
        seg_no = 0
        for step in plan:
            if step["kind"] != "opencv":
                continue
            step["file"] = "seg%d.onnx" % seg_no
            aliases = write_segment(full, os.path.join(args.out, step["file"]),
                                    step["inputs"], step["outputs"])
            step["forward"] = [o + "__out" if o + "__out" in aliases else o for o in step["outputs"]]
            seg_no += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in model.graph.value_info}
    head = {}
    for name in head_outputs:
        stride = args.imgsz // shapes[name][3]
        role = "box" if "/cv2." in name else "cls"
        head.setdefault(stride, {})[role] = name
    manifest = {
        "imgsz": args.imgsz,
        "names": names,
        "reg_max": shapes[head[min(head)]["box"]][1] // 4,
        "head": [dict(stride=s, **head[s]) for s in sorted(head)],
        "steps": plan,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("Classes:", names)
    print("Wrote %s: %d OpenCV segment(s), %d numpy block(s), reg_max=%d"
          % (args.out, seg_no, sum(s["kind"] == "numpy" for s in plan), manifest["reg_max"]))


if __name__ == "__main__":
    main()
