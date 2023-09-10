import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clip_embeddings(example, clip_model, clip_processor):

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
    dataset = dataset.map(lambda example: {"similarity": torch.matmul(torch.tensor(text_embed), torch.tensor(example["image_embeds"]).t()).item()}, num_proc=8)
    dataset = dataset.sort("similarity", reverse=True)
    return dataset


def compute_image_score(results, higher_is_better=True):

    correct_count = 0

    for result in results:

        if higher_is_better:
            correct = result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]
        else:
            correct = result["c0_i0"] < result["c0_i1"] and result["c1_i1"] < result["c1_i0"]

        if correct:
            correct_count += 1

    denominator = len(results)

    return correct_count/denominator


def get_clm_loss(image, text, model, processor):

    inputs = processor(images=image.convert("RGB"), text=text, return_tensors="pt")

    # Don't compute perplexity of the [SEP] token.
    inputs["input_ids"] = inputs["input_ids"][:,:-1]
    inputs["attention_mask"] = inputs["attention_mask"][:,:-1]

    inputs["labels"] = inputs["input_ids"]

    inputs.to(device)

    loss = model(**inputs).loss.item()  # For ranking purposes, the CLM loss is effectively the same as perplexity.

    return loss


def get_itm_score(image, text, model, processor):

    inputs = processor(text=text, images=image.convert("RGB"), return_tensors="pt")

    inputs.to(device)

    itm_score = model(**inputs)[0][0][1].item()
    return itm_score


def get_contrastive_score(image, text, model, processor):

    inputs = processor(text=text, images=image.convert("RGB"), return_tensors="pt")

    inputs.to(device)

    contrastive_score = model(**inputs, use_itm_head=False)[0].item()
    return contrastive_score


def get_most_probable_text_with_swapped_tokens(text, mlm, mlm_tokenizer, freq_filter=0):

    encoded = mlm_tokenizer.encode(text)
    # We assume that the mlm tokenizer adds special characters to the beginning and end.
    acceptable_tokens = set(filter(lambda token: token >= freq_filter, set(encoded[1:-1])))
    acceptable_indices = list(filter(lambda index: encoded[index] in acceptable_tokens, range(len(encoded))))
    batch_size = 1000  # This is the number of swap samples we take.
    configuration_batch = [encoded.copy() for _ in range(batch_size)]
    configuration_batch = torch.tensor(configuration_batch, device=device)
    possible_i1_and_i2 = set()
    for i1 in acceptable_indices:
        for i2 in acceptable_indices:
            if encoded[i1] != encoded[i2]:
                possible_i1_and_i2.add((i1, i2))

    if len(possible_i1_and_i2) < batch_size:
        batch_size = len(possible_i1_and_i2)
        indices1, indices2 = zip(*possible_i1_and_i2)
    else:
        possible_i1_and_i2_as_list = list(possible_i1_and_i2)
        possible_i1_and_i2_list_indices = range(len(possible_i1_and_i2_as_list))
        sampled_list_indices = np.random.choice(possible_i1_and_i2_list_indices, size=batch_size, replace=False)
        indices1, indices2 = zip(*[possible_i1_and_i2_as_list[index] for index in sampled_list_indices])
    tokens1 = [encoded[index] for index in indices1]
    tokens2 = [encoded[index] for index in indices2]
    indices1 = torch.tensor(indices1, device=device)
    indices2 = torch.tensor(indices2, device=device)
    tokens1 = torch.tensor(tokens1, device=device)
    tokens2 = torch.tensor(tokens2, device=device)
    with torch.no_grad():
        configuration_batch[range(batch_size),indices1] = torch.tensor([mlm_tokenizer.mask_token_id]*batch_size, device=device)
        l_i1_t1_given_i2_t2 = mlm(input_ids=torch.tensor(configuration_batch, device=device)).logits[range(batch_size), indices1, tokens1]
        configuration_batch[range(batch_size),indices2] = torch.tensor(tokens1, device=device)
        l_i1_t2_given_i2_t1 = mlm(input_ids=torch.tensor(configuration_batch, device=device)).logits[range(batch_size), indices1, tokens2]
        configuration_batch[range(batch_size),indices2] = torch.tensor([mlm_tokenizer.mask_token_id]*batch_size, device=device)
        configuration_batch[range(batch_size),indices1] = torch.tensor(tokens1, device=device)
        l_i2_t2_given_i1_t1 = mlm(input_ids=torch.tensor(configuration_batch, device=device)).logits[range(batch_size), indices2, tokens2]
        configuration_batch[range(batch_size),indices1] = torch.tensor(tokens2, device=device)
        l_i2_t1_given_i1_t2 = mlm(input_ids=torch.tensor(configuration_batch, device=device)).logits[range(batch_size), indices2, tokens1]
        configuration_batch[range(batch_size),indices2] = torch.tensor(tokens1, device=device)

    l_i2_t1_and_i1_t2 = l_i2_t1_given_i1_t2 + l_i1_t2_given_i2_t1
    l_i1_t1_and_i2_t2 = l_i1_t1_given_i2_t2 + l_i2_t2_given_i1_t1
    probabilities = torch.nn.functional.softmax(torch.cat([l_i1_t1_and_i2_t2.unsqueeze(0), l_i2_t1_and_i1_t2.unsqueeze(0)]), dim=0)
    p_i1_t1_and_i2_t2, p_i2_t1_and_i1_t2 = probabilities[0], probabilities[1]
    highest_probability_swap_index = torch.argmax(p_i2_t1_and_i1_t2 - p_i1_t1_and_i2_t2)
    return mlm_tokenizer.decode(configuration_batch[highest_probability_swap_index].tolist(), skip_special_tokens=True), p_i1_t1_and_i2_t2[highest_probability_swap_index] < p_i2_t1_and_i1_t2[highest_probability_swap_index]
