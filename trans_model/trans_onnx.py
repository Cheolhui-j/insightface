
import os
import onnx

import onnxruntime
import numpy as np
from config import get_config
from Learner import face_learner
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    conf = get_config()

    learner = face_learner(conf)

    # learner.load_state(conf, 'resnet_50_accuracy:0.9534_step:682200_None.pth', model_only=True)  # cutoff
    # learner.load_state(conf, 'resnet_50_accuracy:0.9546_step:1055136_None.pth', model_only=True) # r50
    learner.load_state(conf, 'resnet_101_accuracy:0.9681_step:450252_None.pth', model_only=True) # r100
    # learner.load_state(conf, 'resnet_50_accuracy:0.9666_step:773160_None.pth', model_only=True) # translation

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = learner.model
    model.to(device)
    model.eval()

    batch_size = 1
    x = torch.rand(batch_size, 3, 112, 112, requires_grad=True).to(device)

    # model에서 계산된 결과값
    torch_out = model(x)

    torch.onnx.export(model, x, "./resnet100_cosine.onnx", export_params=True, do_constant_folding=True, verbose=True,input_names=['input'], output_names=['output'] )

    ort_session = onnxruntime.InferenceSession("./resnet100_cosine.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(ort_outs[0].shape)
    print(torch_out.shape)

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")