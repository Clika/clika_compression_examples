import os

import onnx
import torch
from typing import List, Optional, Tuple
import onnxruntime as ort
from ultralytics.utils.tal import dist2bbox, make_anchors
from pathlib import Path


def util_decode_bboxes(outputs: List[torch.Tensor], stride: Optional[torch.Tensor] = None, reg_max: int = 16, num_classes: int = 80):
    # outputs: 3 convolution outputs (end of the weighted layers)
    # stride: YOLO(model_name).model.stride
    # reg_max: 16
    # num_classes: YOLO(model_name).model.nc
    if stride is None:
        stride = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = make_anchors(outputs, stride, 0.5)
    anchors = anchors.T
    strides = strides.T
    bsz, _, _, _ = outputs[0].shape  # BCHW

    no = num_classes + reg_max * 4
    x_cat = torch.cat([xi.view(bsz, no, -1) for xi in outputs], 2)
    box, cls = x_cat.split((reg_max * 4, num_classes), 1)

    if reg_max > 1:
        _dfl_kernel = torch.arange(reg_max, device=outputs[0].device, dtype=outputs[0].dtype)
        _dfl_kernel = _dfl_kernel.view(1, -1, 1, 1)
        box = box.view(bsz, 4, reg_max, -1).transpose(2, 1).softmax(1)
        box = torch.nn.functional.conv2d(box, _dfl_kernel)
        box = box.view(bsz, 4, -1)
    dbox = dist2bbox(box, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    return torch.cat((dbox, cls.sigmoid()), 1)


class PostProcessing(torch.nn.Module):

    def __init__(self, stride: Optional[list], is_detection: bool, is_export_mode: bool):
        super().__init__()
        if stride is None:
            stride = [8.0, 16.0, 32.0]
        self.is_detection = is_detection
        self.is_export_mode = is_export_mode
        self.stride = torch.tensor(stride)

    def forward(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        x = outputs[:3]
        bboxes = util_decode_bboxes(x, stride=self.stride, reg_max=16, num_classes=80)
        if self.is_detection:
            final_outputs: list = [bboxes]
            if not self.is_export_mode:
                final_outputs.append(outputs)
            return tuple(final_outputs)
        else:
            mc, p = outputs[3:]
            final_outputs: list = [bboxes]
            if self.is_export_mode:
                final_outputs.append(p)
            else:
                final_outputs.extend([*x, mc, p])
            return tuple(final_outputs)


@torch.no_grad()
def merge_post_processing_to_onnx(model_onnx_path: Path, final_save_path: Path, is_detection_model: bool, is_export_mode: bool):
    stride = [8.0, 16.0, 32.0]
    dummy_xs = torch.rand(1, 3, 640, 640)
    session = ort.InferenceSession(model_onnx_path)
    outputs = session.run(None, {"xs": dummy_xs.numpy()})
    outputs = [torch.from_numpy(x) for x in outputs]
    temp_path = model_onnx_path.with_stem(f"{model_onnx_path.stem}_temp")

    pp_input_names: list = [f"features_{i}" for i in range(len(outputs))]
    pp_output_names: list = ["detections"]
    if is_detection_model:
        if not is_export_mode:
            pp_output_names.extend(pp_input_names)
    else:
        if is_export_mode:
            pp_output_names.append(pp_input_names[-1])
        else:
            pp_output_names.extend(pp_input_names)
    dynamic_axes: dict = {
        pp_input_names[i]: {
            j: f"{pp_input_names[i]}_{j}" for j in range(outputs[i].ndim)
        } for i in range(len(pp_input_names))
    }
    torch.onnx.export(
        PostProcessing(stride=stride, is_detection=is_detection_model, is_export_mode=is_export_mode),
        (outputs,),
        f=str(temp_path),
        dynamic_axes=dynamic_axes,
        input_names=pp_input_names,
        output_names=pp_output_names,
        dynamo=False,
        do_constant_folding=True,
        opset_version=20
    )
    try:
        import onnxsim
        modified_pp, check_ok = onnxsim.simplify(onnx.load(temp_path))
        onnx.save(modified_pp, temp_path)
    except:
        pass

    # merging time
    yolo_model = onnx.load(model_onnx_path)
    post_processing_model = onnx.load(temp_path)

    # HACKY ZONE:
    post_processing_model.ir_version = 10  # HACK, onnx upgrade path for opset is broken.
    while len(post_processing_model.opset_import) > 0:
        post_processing_model.opset_import.pop()
    for opset_import in yolo_model.opset_import:
        post_processing_model.opset_import.append(opset_import)
    # END OF HACKY ZONE

    yolo_input_names: List[str] = [x.name for x in list(yolo_model.graph.input)]
    yolo_output_names: List[str] = [x.name for x in list(yolo_model.graph.output)]

    pp_input_names: List[str] = [x.name for x in list(post_processing_model.graph.input)]
    pp_output_names: List[str] = [x.name for x in list(post_processing_model.graph.output)]

    final_model = onnx.compose.merge_models(
        yolo_model, post_processing_model,
        list(zip(yolo_output_names, pp_input_names)),
        inputs=yolo_input_names,
        outputs=pp_output_names,
        producer_name="clika.io",
        doc_string=model_onnx_path.name,
    )
    onnx.save(final_model, f=final_save_path)
    os.remove(temp_path)
    try:
        onnx.shape_inference.infer_shapes_path(final_save_path, final_save_path)
    except:
        pass
