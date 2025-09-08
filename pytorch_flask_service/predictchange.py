import argparse
import torch
import torchvision.transforms as transforms
import pickle

from PIL import Image

from utils.load_data import get_loader
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder
from metrics import compute_metrics, generate_text_file, generate_result
from IUdata.build_vocab import JsonReader, Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
def predictchange(fname, args):
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
    test_transforms = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    # testing dataset loader
    # eval_data_loader = get_loader(args.image_dir, args.eval_json_dir,
    #                               vocab, test_transforms, args.eval_batch_size,
    #                               args.num_workers, args.max_impression_len,
    #                               args.max_sen_num, args.max_single_sen_len, shuffle=False)

    print("1")
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
    print("2")
    # image_encoder.load_state_dict(torch.load(img_en_path))
    impression_decoder.load_state_dict(torch.load(imp_de_path,map_location=torch.device('cpu')))
    finding_decoder.load_state_dict(torch.load(fin_de_path,map_location=torch.device('cpu')))

    # Generate impressions and findings
    pre_imps_lst, pre_fins_lst = [],[]
    image_path = fname
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image)
    image = torch.reshape(image,(1,3,224,224))
    # print(image)
    frontal_imgs = image
    # print(frontal_imgs.shape)
    global_feas = image_encoder(frontal_imgs)
    global_feas = torch.reshape(global_feas,(1,2048))
    # print(global_feas.shape)
    print("3")
    predicted_imps, global_topic_vec = impression_decoder.sampler(global_feas, args.max_impression_len)
    # print(predicted_imps)
    # print(global_topic_vec)
    # global_topic_vec = torch.unsqueeze(global_topic_vec, dim=0)
    # print(global_topic_vec)
    print("4")
    predicted_fins = finding_decoder.sampler(global_feas, global_topic_vec, args.max_single_sen_len,
                                                 args.max_sen_num)
    pre_imps_lst = predicted_imps
    pre_fins_lst = predicted_fins
    pre_imps, pre_fins = generate_result(pre_imps_lst, pre_fins_lst, args)
    return pre_imps, pre_fins