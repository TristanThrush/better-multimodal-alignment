from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from utils import device, get_clm_loss, get_contrastive_score, get_itm_score, compute_image_score, get_most_probable_text_with_swapped_tokens

mlm_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
mlm = AutoModelForMaskedLM.from_pretrained('distilroberta-base').to("cuda")

blip_clm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_clm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
blip_clm_model.eval()

blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
blip_itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
blip_itm_model.eval()

winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]

clm_losses = []
itm_scores = []
contrastive_scores = []
for example in tqdm(winoground):

    image_0 = example["image_0"]
    image_1 = example["image_1"]
    text_0 = example["caption_0"]
    alt_text_0 = get_most_probable_text_with_swapped_tokens(example["caption_0"], mlm, mlm_tokenizer)
    text_1 = example["caption_1"]
    alt_text_1 = get_most_probable_text_with_swapped_tokens(example["caption_1"], mlm, mlm_tokenizer)


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


