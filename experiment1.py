from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, CLIPModel, CLIPProcessor
from datasets import load_dataset
from utils import device, get_clm_loss, get_contrastive_score, get_itm_score, compute_image_score
import argparse
import torch
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--eval_winoground", action="store_true")
parser.add_argument("--eval_flickr30k_ir", action="store_true")
args = parser.parse_args()

blip_clm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_clm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
blip_clm_model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
blip_itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
blip_itm_model.eval()

winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]
flickr30k_test = load_dataset("Tristan/flickr30k_test", use_auth_token=True)["test"]

def clip_embeddings(example):

    text_inputs = clip_processor(text=example["caption"], padding=True, truncation=True, return_tensors="pt")
    text_inputs.to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    image_inputs = clip_processor(images=example["image"], return_tensors="pt")
    image_inputs.to(device)
    image_features = clip_model.get_image_features(**image_inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return {"image_embeds": image_features.to("cpu").detach(), "text_embeds": text_features.to("cpu").detach()}

def get_clip_top_images(text_embed, dataset):
    dataset = dataset.map(lambda example: {"similarity": torch.matmul(torch.tensor(text_embed), torch.tensor(example["image_embeds"]).t()).item()})
    dataset = dataset.sort("similarity", reverse=True)
    return dataset

flickr30k_test = flickr30k_test.map(lambda example: clip_embeddings(example))

if args.eval_flickr30k_ir:
    clm_r1 = []
    itm_r1 = []
    contrastive_r1 = []
    no_reranking_r1 = []
    for example in tqdm(flickr30k_test):
        ground_truth_img_id = example["img_id"]
        for caption_idx in range(len(example["caption"])):
            caption = example["caption"][caption_idx]
            text_embed = example["text_embeds"][caption_idx]
            min_clm_loss = None
            clm_img_id = None
            max_itm_score = None
            itm_img_id = None
            max_contrastive_score = None
            contrastive_img_id = None
            top_5 = get_clip_top_images(text_embed, flickr30k_test).select(range(5))
            for example2 in top_5:
                clm_loss = get_clm_loss(example2["image"], caption, blip_clm_model, blip_clm_processor)
                if min_clm_loss is None or min_clm_loss > clm_loss:
                    min_clm_loss = clm_loss
                    clm_img_id = example2["img_id"]
                itm_score = get_itm_score(example2["image"], caption, blip_itm_model, blip_itm_processor)
                if max_itm_score is None or max_itm_score < itm_score:
                    max_itm_score = itm_score
                    itm_img_id = example2["img_id"]
                contrastive_score = get_contrastive_score(example2["image"], caption, blip_itm_model, blip_itm_processor)
                if max_contrastive_score is None or max_contrastive_score < contrastive_score:
                    max_contrastive_score = contrastive_score
                    contrastive_img_id = example2["img_id"]
            clm_r1.append(int(ground_truth_img_id == clm_img_id))
            itm_r1.append(int(ground_truth_img_id == itm_img_id))
            contrastive_r1.append(int(ground_truth_img_id == contrastive_img_id))
            no_reranking_r1.append(int(ground_truth_img_id == top5[0]["img_id"]))
    print("Flickr30k test R@1 with no reranking", statistics.mean(no_reranking_r1))
    print("Flickr30k test R@1 with BLIP CLM head top 5 reranking", statistics.mean(clm_r1))
    print("Flickr30k test R@1 with BLIP ITM head top 5 reranking", statistics.mean(itm_r1))
    print("Flickr30k test R@1 with BLIP Contrastive head top 5 reranking", statistics.mean(contrastive_r1))


if args.eval_winoground:
    clm_losses = []
    itm_scores = []
    contrastive_scores = []
    for example in tqdm(winoground):

        image_0 = example["image_0"]
        image_1 = example["image_1"]
        text_0 = example["caption_0"]
        text_1 = example["caption_1"]


        clm_losses.append({
           "id" : example["id"],
           "c0_i0": get_clm_loss(image_0, text_0, blip_clm_model, blip_clm_processor),
           "c0_i1": get_clm_loss(image_1, text_0, blip_clm_model, blip_clm_processor),
          "c1_i0": get_clm_loss(image_0, text_1, blip_clm_model, blip_clm_processor),
          "c1_i1": get_clm_loss(image_1, text_1, blip_clm_model, blip_clm_processor),
        })

        itm_scores.append({
            "id" : example["id"],
             "c0_i0": get_itm_score(image_0, text_0, blip_itm_model, blip_itm_processor),
             "c0_i1": get_itm_score(image_1, text_0, blip_itm_model, blip_itm_processor),
             "c1_i0": get_itm_score(image_0, text_1, blip_itm_model, blip_itm_processor),
             "c1_i1": get_itm_score(image_1, text_1, blip_itm_model, blip_itm_processor),
        })

        contrastive_scores.append({
            "id" : example["id"],
            "c0_i0": get_contrastive_score(image_0, text_0, blip_itm_model, blip_itm_processor),
            "c0_i1": get_contrastive_score(image_1, text_0, blip_itm_model, blip_itm_processor),
            "c1_i0": get_contrastive_score(image_0, text_1, blip_itm_model, blip_itm_processor),
            "c1_i1": get_contrastive_score(image_1, text_1, blip_itm_model, blip_itm_processor),
        })


    print("Image Score (P(T | I) from BLIP CLM Head):", compute_image_score(clm_losses, higher_is_better=False))
    print("Image Score (BLIP ITM Head):", compute_image_score(itm_scores))
    print("Image Score (BLIP Contrastive Head):", compute_image_score(contrastive_scores))

