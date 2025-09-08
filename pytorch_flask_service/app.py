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

from metrics import generate_result
# from model import MobileNetV2
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder
from build_vocab import Vocabulary
app = Flask(__name__)
CORS(app)
# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(img_en_path, imp_de_path, fin_de_path, image, args):
    """"load trained models and generate impressions and findings for evaluating"""
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    # Models
    image_encoder = EncoderCNN().eval().to(device)
    impression_decoder = Impression_Decoder(args.embed_size, args.hidden_size,
                                                 vocab_size, args.imp_layers_num,
                                                 args.num_global_features, args.num_conv1d_out,
                                                 args.teach_rate, args.max_impression_len).eval().to(device)
    finding_decoder = Atten_Sen_Decoder(args.embed_size, args.hidden_size, vocab_size,
                                             args.fin_num_layers, args.sen_enco_num_layers,
                                             args.num_global_features, args.num_regions, args.num_conv1d_out,
                                             args.teach_rate, args.max_single_sen_len, args.max_sen_num).eval().to(device)
    # load trained model weights
    image_encoder.load_state_dict(torch.load(img_en_path, map_location=torch.device('cpu')))

    # image_encoder.load_state_dict(torch.load(img_en_path))
    impression_decoder.load_state_dict(torch.load(imp_de_path,map_location=torch.device('cpu')))
    finding_decoder.load_state_dict(torch.load(fin_de_path,map_location=torch.device('cpu')))

    # Generate impressions and findings
    pre_imps_lst, pre_fins_lst = [],[]
    # print(image)
    frontal_imgs = image
    global_feas = image_encoder(frontal_imgs)
    global_feas = torch.reshape(global_feas,(1,2048))
    predicted_imps, global_topic_vec = impression_decoder.sampler(global_feas, args.max_impression_len)
    predicted_fins = finding_decoder.sampler(global_feas, global_topic_vec, args.max_single_sen_len,
                                                 args.max_sen_num)
    pre_imps_lst = predicted_imps
    pre_fins_lst = predicted_fins

    return pre_imps_lst, pre_fins_lst
def transform_image(image_bytes):
    test_transforms = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = test_transforms(image)
    image = torch.reshape(image,(1,3,224,224))
    return image
def get_prediction(image_bytes):
    try:
        input = transform_image(image_bytes=image_bytes)
        predicted_imps_lst, predicted_fins_lst  = test(img_en_path, imp_de_path, fin_de_path, input, args)
        pre_imp_dic, pre_fin_dic = generate_result(predicted_imps_lst, predicted_fins_lst, args)
            # 处理前端传递的数据
            # 调用 test 函数生成印象和发现
            # 返回预测结果
        output = {'impression': pre_imp_dic, 'finding': pre_fin_dic}
        return_info = {"result": output}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info
@app.route('/favicon.ico')
def favicon():
    return '', 204
@app.route('/predict', methods=['POST'])
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
    parser.add_argument('--model_path', type=str, default='./',
                        help='path for weights')
    parser.add_argument('--vocab_path', type=str, default='./IUdata_vocab_0threshold.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal', help='directory for X-ray images')
    parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json', help='the path for json file')
    # model parameters
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size for loading data')
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
    #################################################################################################################################
    #################################################################################################################################

    args = parser.parse_args()
    print(args)
    img_en_path, imp_de_path, fin_de_path = None, None, None
    num_ckpt = 0
    for path in os.listdir(args.model_path):
        if path.split(".")[-1] == 'ckpt':
            num_ckpt += 1
            if 'image' in path:
                img_en_path = os.path.join(args.model_path, path)
            elif 'impression' in path:
                imp_de_path = os.path.join(args.model_path, path)
            elif 'finding' in path:
                fin_de_path = os.path.join(args.model_path, path)
    app.run(host="0.0.0.0", port=5000)