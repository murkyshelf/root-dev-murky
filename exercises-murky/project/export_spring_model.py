import torch
from spring_pinn import SpringNet

model = SpringNet()
model.load_state_dict(torch.load("spring_pinn_weights.pth"))
model.eval()
dummy_input = torch.randn(1,1)

# export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "spring_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)

print("Model exported to spring_model.onnx")