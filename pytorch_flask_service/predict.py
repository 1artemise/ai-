import os
import io
import json
import torch
import argparse
import pickle
import os
import torchvision.transforms as transforms
from metrics import generate_result
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder
from build_vocab import Vocabulary
def test(img_en_path, imp_de_path, fin_de_path, image, args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)
    image_encoder = EncoderCNN()
    impression_decoder = Impression_Decoder(args.embed_size, args.hidden_size,
                                            vocab_size, args.imp_layers_num,
                                            args.num_global_features, args.num_conv1d_out,
                                            args.teach_rate, args.max_impression_len)
    finding_decoder = Atten_Sen_Decoder(args.embed_size, args.hidden_size, vocab_size,
                                        args.fin_num_layers, args.sen_enco_num_layers,
                                        args.num_global_features, args.num_regions, args.num_conv1d_out,
                                        args.teach_rate, args.max_single_sen_len, args.max_sen_num)

    # load trained model weights
    image_encoder.load_state_dict(torch.load(img_en_path, map_location=torch.device('cpu')))
    impression_decoder.load_state_dict(torch.load(imp_de_path, map_location=torch.device('cpu')))
    finding_decoder.load_state_dict(torch.load(fin_de_path, map_location=torch.device('cpu')))

    # Generate impressions and findings
    pre_imps_lst, pre_fins_lst = [],[]
    # print(image)
    print(img_en_path)
    frontal_imgs = image
    global_feas = image_encoder(frontal_imgs)
    global_feas = torch.reshape(global_feas,(1,2048))
    predicted_imps, global_topic_vec = impression_decoder.sampler(global_feas, args.max_impression_len)
    print("predicted_imps",predicted_imps)
    predicted_fins = finding_decoder.sampler(global_feas, global_topic_vec, args.max_single_sen_len,
                                                 args.max_sen_num)
    pre_imps_lst = predicted_imps
    pre_fins_lst = predicted_fins
    return pre_imps_lst, pre_fins_lst
def transform_pic(image,args):
    print("transform_pic")
    test_transforms = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = test_transforms(image)
    image = torch.reshape(image, (1, 3, 224, 224))
    return image
def predict(img,args):
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
    print(img_en_path)
    img2 = transform_pic(img,args)
    print(img)
    num_run = "test"
    predicted_imps_lst, predicted_fins_lst = test(img_en_path, imp_de_path, fin_de_path, img2, args)

    pre_imps, pre_fins = generate_result(predicted_imps_lst, predicted_fins_lst, args)
    print("pre_imps",pre_imps)
    print("pre_fins",pre_fins)
    return pre_imps, pre_fins