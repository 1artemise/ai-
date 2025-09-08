import os
import io
import json
import torch
import argparse
import pickle
import os
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
# from model import MobileNetV2
from utils.load_data import get_loader
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder

app = Flask(__name__)
CORS(app)  # 解决跨域问题

weights_path = "./MobileNetV2(flower).pth"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
model = MobileNetV2(num_classes=5).to(device)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval()

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--model_path', type=str, default='model_weights', help='path for weights')
    parser.add_argument('--vocab_path', type=str, default='IUdata/IUdata_vocab_0threshold.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal', help='directory for X-ray images')
    parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json', help='the path for json file')
    # model parameters
    parser.add_argument('--eval_batch_size', type=int, default=75, help='batch size for loading data')
    parser.add_argument('--num_workers', type=int, default=2, help='multi-process data loading')
    parser.add_argument('--max_impression_len', type=int, default=15,
                        help='The maximum length of the impression (one or several sentences)')
    parser.add_argument('--max_single_sen_len', type=int, default=15,
                        help='The maximum length of the each sentence in the finding')
    parser.add_argument('--max_sen_num', type=int, default=7, help='The maximum number of sentences in the finding')
    parser.add_argument('--single_punc', type=bool, default=True,
                        help='Take punctuation as a single word: If true, generate sentences such as: Hello , world .')
    parser.add_argument('--imp_fin_only', type=bool, default=False, help='Only evaluate on Impression+Finding')

    #################################################################################################################################
    #################################################################################################################################
    # not changed parameters
    parser.add_argument('--resize_size', type=int, default=256, help='The resize size of the X-ray image')
    parser.add_argument('--crop_size', type=int, default=224, help='The crop size of the X-ray image')
    parser.add_argument('--embed_size', type=int, default=512, help='The embed_size for vocabulary and images')
    parser.add_argument('--hidden_size', type=int, default=512, help='The number of hidden states in LSTM layers')
    parser.add_argument('--num_global_features', type=int, default=2048,
                        help='The number of global features for image encoder')
    parser.add_argument('--imp_layers_num', type=int, default=1, help='The number of LSTM layers in impression decoder')
    parser.add_argument('--fin_num_layers', type=int, default=2, help='The number of LSTM layers in finding decoder ')
    parser.add_argument('--sen_enco_num_layers', type=int, default=3,
                        help='The number of convolutional layer in topic encoder')
    parser.add_argument('--num_local_features', type=int, default=2048,
                        help='The channel number of local features for image encoder')
    parser.add_argument('--num_regions', type=int, default=49, help='The number of sub-regions for local features')
    parser.add_argument('--num_conv1d_out', type=int, default=1024,
                        help='The number of output channels for 1d convolution of sentence encoder')
    parser.add_argument('--teach_rate', type=float, default=0.0, help='No teach force is used in testing')
    parser.add_argument('--log_step', type=int, default=100, help='The interval of displaying the loss and perplexity')
    parser.add_argument('--save_step', type=int, default=1000, help='The interval of saving weights of models')
    app.run(host="0.0.0.0", port=5000)




