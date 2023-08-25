from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval
from datasets import load_dataset
from utils import device, get_clm_loss, get_contrastive_score, get_itm_score, compute_image_score

blip_clm_processor = BlipProcessor.from_pretrained("Salseforce/blip-image-captioning-large")
blip_clm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
blip_clm_model.eval()

blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
blip_itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
blip_itm_model.eval()

winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]

for example in tqdm(winoground):

    image_0 = example["image_0"]
    image_1 = example["image_1"]
    text_0 = example["caption_0"]
    text_1 = example["caption_1"]


    losses = {
        "id" : example["id"],
        "c0_i0": get_clm_loss(image_0, text_0, blip_clm_model, blip_clm_processor),
        "c0_i1": get_clm_loss(image_1, text_0, blip_clm_model, blip_clm_processor),
        "c1_i0": get_clm_loss(image_0, text_1, blip_clm_model, blip_clm_processor),
        "c1_i1": get_clm_loss(image_1, text_1, blip_clm_model, blip_clm_processor),
    }
    print("Image Score (P(T | I) from BLIP CLM Head):", compute_image_score(losses, higher_is_better=False))

    itm_scores = {
        "id" : example["id"],
        "c0_i0": get_itm_score(image_0, text_0, blip_itm_model, blip_itm_processor),
        "c0_i1": get_itm_score(image_1, text_0, blip_itm_model, blip_itm_processor),
        "c1_i0": get_itm_score(image_0, text_1, blip_itm_model, blip_itm_processor),
        "c1_i1": get_itm_score(image_1, text_1, blip_itm_model, blip_itm_processor),
    }
    print("Image Score (BLIP ITM Head):", compute_image_score(itm_scores))

    contrastive_scores = {
        "id" : example["id"],
        "c0_i0": get_contrastive_score(image_0, text_0, blip_itm_model, blip_itm_processor),
        "c0_i1": get_contrastive_score(image_1, text_0, blip_itm_model, blip_itm_processor),
        "c1_i0": get_contrastive_score(image_0, text_1, blip_itm_model, blip_itm_processor),
        "c1_i1": get_contrateive_score(image_1, text_1, blip_itm_model, blip_itm_processor),
    }
    print("Image Score (BLIP Contrastive Head):", compute_image_score(contrastive_scores))

