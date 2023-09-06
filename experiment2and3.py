from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, AutoTokenizer, AutoModelForMaskedLM, CLIPModel, CLIPProcessor
from datasets import load_dataset
from utils import device, get_clm_loss, get_contrastive_score, get_itm_score, compute_image_score, get_most_probable_text_with_swapped_tokens, clip_embeddings, get_clip_top_images
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--eval_winoground", action="store_true")
parser.add_argument("--eval_flickr30k_ir", action="store_true")
args = parser.parse_args()

mlm_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
mlm = AutoModelForMaskedLM.from_pretrained('distilroberta-base').to("cuda")

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

flickr30k_test = flickr30k_test.map(lambda example: clip_embeddings(example, clip_model, clip_processor))

if args.eval_flickr30k_ir:
    clm_r1 = []
    itm_r1 = []
    contrastive_r1 = []
    no_reranking_r1 = []
    for example in tqdm(flickr30k_test):
        ground_truth_img_id = example["img_id"]
        for caption_idx in range(len(example["caption"])):
            caption = example["caption"][caption_idx]
            alt_caption, use_alt_caption = get_most_probable_text_with_swapped_tokens(caption, mlm, mlm_tokenizer, freq_filter=1000)
            print("use_alt_caption", use_alt_caption)
            text_embed = example["text_embeds"][caption_idx]
            min_clm_loss = None
            clm_img_id = None
            max_itm_score = None
            itm_img_id = None
            max_contrastive_score = None
            contrastive_img_id = None
            top_5 = get_clip_top_images(text_embed, flickr30k_test).select(range(5))
            for example2 in top_5:
                if use_alt_caption:
                    clm_loss = get_clm_loss(example2["image"], caption, blip_clm_model, blip_clm_processor) / get_clm_loss(example2["image"], alt_caption, blip_clm_model, blip_clm_processor)
                else:
                    clm_loss = get_clm_loss(example2["image"], caption, blip_clm_model, blip_clm_processor)

                if min_clm_loss is None or min_clm_loss > clm_loss:
                    min_clm_loss = clm_loss
                    clm_img_id = example2["img_id"]

                if use_alt_caption:
                    itm_score = get_itm_score(example2["image"], caption, blip_itm_model, blip_itm_processor) / get_itm_score(example2["image"], alt_caption, blip_itm_model, blip_itm_processor)
                else:
                    itm_score = get_itm_score(example2["image"], caption, blip_itm_model, blip_itm_processor)

                if max_itm_score is None or max_itm_score < itm_score:
                    max_itm_score = itm_score
                    itm_img_id = example2["img_id"]

                if use_alt_caption:
                    contrastive_score = get_contrastive_score(example2["image"], caption, blip_itm_model, blip_itm_processor) / get_contrastive_score(example2["image"], alt_caption, blip_itm_model, blip_itm_processor)
                else:
                    contrastive_score = get_contrastive_score(example2["image"], caption, blip_itm_model, blip_itm_processor)

                if max_contrastive_score is None or max_contrastive_score < contrastive_score:
                    max_contrastive_score = contrastive_score
                    contrastive_img_id = example2["img_id"]
            clm_r1.append(int(ground_truth_img_id == clm_img_id))
            itm_r1.append(int(ground_truth_img_id == itm_img_id))
            contrastive_r1.append(int(ground_truth_img_id == contrastive_img_id))
            no_reranking_r1.append(int(ground_truth_img_id == top_5[0]["img_id"]))
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
        alt_text_0 = get_most_probable_text_with_swapped_tokens(example["caption_0"], mlm, mlm_tokenizer)[0]
        text_1 = example["caption_1"]
        alt_text_1 = get_most_probable_text_with_swapped_tokens(example["caption_1"], mlm, mlm_tokenizer)[0]


        clm_losses.append({
            "id" : example["id"],
            "c0_i0": get_clm_loss(image_0, text_0, blip_clm_model, blip_clm_processor) / get_clm_loss(image_0, alt_text_0, blip_clm_model, blip_clm_processor),
            "c0_i1": get_clm_loss(image_1, text_0, blip_clm_model, blip_clm_processor) / get_clm_loss(image_1, alt_text_0, blip_clm_model, blip_clm_processor),
            "c1_i0": get_clm_loss(image_0, text_1, blip_clm_model, blip_clm_processor) / get_clm_loss(image_0, alt_text_1, blip_clm_model, blip_clm_processor),
            "c1_i1": get_clm_loss(image_1, text_1, blip_clm_model, blip_clm_processor) / get_clm_loss(image_1, alt_text_1, blip_clm_model, blip_clm_processor),
        })

        itm_scores.append({
            "id" : example["id"],
            "c0_i0": get_itm_score(image_0, text_0, blip_itm_model, blip_itm_processor) / get_itm_score(image_0, alt_text_0, blip_itm_model, blip_itm_processor),
            "c0_i1": get_itm_score(image_1, text_0, blip_itm_model, blip_itm_processor) / get_itm_score(image_1, alt_text_0, blip_itm_model, blip_itm_processor),
            "c1_i0": get_itm_score(image_0, text_1, blip_itm_model, blip_itm_processor) / get_itm_score(image_0, alt_text_1, blip_itm_model, blip_itm_processor),
            "c1_i1": get_itm_score(image_1, text_1, blip_itm_model, blip_itm_processor) / get_itm_score(image_1, alt_text_1, blip_itm_model, blip_itm_processor),
        })

        contrastive_scores.append({
            "id" : example["id"],
            "c0_i0": get_contrastive_score(image_0, text_0, blip_itm_model, blip_itm_processor) / get_contrastive_score(image_0, alt_text_0, blip_itm_model, blip_itm_processor),
            "c0_i1": get_contrastive_score(image_1, text_0, blip_itm_model, blip_itm_processor) / get_contrastive_score(image_1, alt_text_0, blip_itm_model, blip_itm_processor),
            "c1_i0": get_contrastive_score(image_0, text_1, blip_itm_model, blip_itm_processor) / get_contrastive_score(image_0, alt_text_1, blip_itm_model, blip_itm_processor),
            "c1_i1": get_contrastive_score(image_1, text_1, blip_itm_model, blip_itm_processor) / get_contrastive_score(image_1, alt_text_1, blip_itm_model, blip_itm_processor),
        })


    print("Image Score (DistilRoBERTa + BLIP CLM Head score ratios):", compute_image_score(clm_losses, higher_is_better=False))
    print("Image Score (DistilRoBERTa + BLIP ITM Head score ratios):", compute_image_score(itm_scores))
    print("Image Score (DistilRoBERTa + BLIP Contrastive Head score ratios):", compute_image_score(contrastive_scores))


