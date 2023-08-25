# Better multimodal alignment scores: A few experiments

Using the code in this repo, we experiment with new ways to extract ITM scores from existing models. With the code, you can find that a model with only a few hundred million parameters gets above the current state of the art on the Winoground Image metric. It surpasses [Google's 17B PaLI](https://arxiv.org/abs/2209.06794) by 10.50 points, even in the most difficult comparison setting ([where the PaLI has been finetuned / prompted to do well at Winoground](https://arxiv.org/abs/2305.10400)).

To follow along with this readme, install the repo first:

```
git clone https://github.com/TristanThrush/better-multimodal-alignment.git
pip install -r requirements.txt
```

Then run the experiment that you are interested in, for example:

```
python experiment1.py
```

## Intro

Multimodal language models are usually pretrained with several objectives, and image-text matching is typically one of them. An image-text matching (ITM) head can be used in downstream applications such as image and text retrieval. But we do not know of an ITM approach that is good at fine-grained image-text alignment. Let’s look an at example:

| <img src="https://datasets-server.huggingface.co/assets/facebook/winoground/--/default/test/14/image_0/image.jpg" width="250" height="250" />               | <img src="https://datasets-server.huggingface.co/assets/facebook/winoground/--/default/test/14/image_1/image.jpg" width="250" height="250" /> |
| - | - |
| a mug in some grass | some grass in a mug |

Popular ITM models tend to know that there is a mug and grass in both images. But they do not understand how the word order of “mug”, “grass” and “in” affects the meaning. They cannot reliably match the captions with the correct images here. The [Winoground evaluation dataset](https://arxiv.org/abs/2204.03162) reveals that many contemporary models cannot do much better than chance at this task! This includes popular models like [FLAVA](https://arxiv.org/abs/2112.04482), [CLIP](https://arxiv.org/abs/2103.00020), and [BLIP](https://arxiv.org/abs/2201.12086). Even the closed-source models that have been tested, such as the 17B parameter [PaLI](https://arxiv.org/abs/2209.06794), do a bit better than chance with [special tuning and prompting](https://arxiv.org/abs/2305.10400) but still not very well. Models have a particularly hard time with the Winoground Image metric, which tests the scenario where a model is given a caption and must pick one of two images. They often perform worse than random chance on this metric because they always rank one of the images as more likely regardless of the caption.

## Experiment 1: Image-conditioned perplexity (no negative training samples needed!)

To learn that an image and text match, ITM models are typically trained with image-caption pairs scraped from the internet. To learn that an image and text don’t match, ITM models are given negative examples. The negatives are usually [randomly paired images and captions](https://arxiv.org/abs/2103.00020) from the same internet dataset, but sometimes more [sophisticated methods](https://arxiv.org/abs/2201.12086) are used. With this standard ITM approach, we would need negative training examples like “grass in mug” versus “mug in grass” paired with the corresponding wrong images above. This would occur extremely infrequently. And generating negative pairs of any kind is a sort of modeling or data augmentation decision that does not come from the natural dataset itself. Luckily, we can extract an ITM score in a way that does not require training with these artificial negative pairings at all.

Given an input Image $I$, and text $T$ made from a sequence of text tokens $t_0,...,t_n$, the probability of the text given the image is:

$P(T | I) = P(t_0, ..., t_n | I) = P(t_0 | I) P(t_1 | I, t_0) ... P(t_n | I, t_0, ..., t_{n-1})$

We got that last expression from the chain rule. Notice that [BLIP](https://arxiv.org/abs/2201.12086)’s causal language modelling (CLM) pretraining objective is designed to give us the probability of the next token given the previous tokens and the image. So we can compute $P(T | I)$ directly. We can also raise this probability to the power of $-1/n$ to get the perplexity of the text given the image in use cases where we need to normalize for the sequence length.

What happens if we use $P(T | I)$ from BLIP's CLM head as our ITM score instead of BLIP’s canonical ITM methods? Overall, we get a better Winoground Image Score. It is still quite close to random chance, but at least there does not seem to be a bias to perform worse than random like the canonical ITM methods.

| Model                          | Image Score  |
|------------------------------- | ------------ |
| Random Chance                  | 25.00        |
| BLIP Contrastive Head          | 16.00        |
| BLIP ITM Head                  | 24.25        |
| $P(T \| I)$ from BLIP CLM Head | **29.00**    |

But $P(T | I)$ isn't quite what we want. For an ITM match score, we really want $P(T, I)$. The normal ITM methods gives us an approximation of $P(T, I)$, but it is poor and relies on artificially-paired negative examples. In the Experiment 3, we will see how to extract some information about $P(T, I)$ from the CLM head of BLIP.

## Experiment 2: Alternate word-order score ratios

Another idea: Is there a way to "normalize" for the contribution that an image adds to the ITM score alone? We want an ITM model to not prefer one image over others independent of the text. For very fine-grained tasks like Winoground, maybe the ITM score should instead be $\frac{P(T, I)}{P(T', I)}$ where $T'$ is a piece of text with the same tokens as $T$, but in a very plausible different order. For example, let's assume that we have an image of "a person wearing a **yellow** dress walking in a field of **red** flowers". We can recognize that "a person wearing a **red** dress walking in a field of **yellow** flowers" is also a very plausible statement, independent of any images. By computing $\frac{P(T, I)}{P(T', I)}$ instead of just $P(T, I)$, we are performing a sort of normalization for the independent effects of an image on an ITM score.

How do we get a plausible alternative caption, given an initial caption, though? For this experiment, we use a small unimodal model which is less than 100M parameters: [DistilRoBERTa](https://arxiv.org/abs/1910.01108). We take many samples of two token swaps at a time and use the MLM probabilities from DistilRoBERTa to get the most probable text with an alternative token order.

Using this method, we see a big improvement in the contrastive head of BLIP, although for some reason not the ITM Head. It is possible that the ITM head's approximation of $P(T, I)$ is so poor that we aren't getting much benefit.

| Model                                              | Image Score  |
|--------------------------------------------------- | ------------ |
| Random Chance                                      | 25.00        |
| BLIP Contrastive Head                              | 16.00        |
| BLIP ITM Head                                      | 24.25        |
| DistilRoBERTa + BLIP Contrastive Head score ratios | **52.00**    |
| DistilRoBERTa + BLIP ITM Head score ratios         | 25.00        |
| PaLI 17B ([with best known finetuning / prompting approach for Winoground](https://arxiv.org/abs/2305.10400)) | 41.50    |
| VQ2 ([with best known finetuning / prompting approach for Winoground](https://arxiv.org/abs/2305.10400)) | 42.20 |

## Experiment 3: No negative training samples + alternate word-order score ratios

Now let's apply both Experiments 1 and 2! In Experiment 1, we figured out how to compute $P(T | I)$ from a CLM head of a model which doesn't need negative training pairs. But we were sad because we really wanted an approximation for $P(T, I)$. And in Experiment 2, we found that it could be even better to normalize for the independent affects of the image by using $\frac{P(T, I)}{P(T', I)}$ as the score.

Even if we can't get $P(T, I)$ from the CLM head, can we at least use it to get the ratio $\frac{P(T, I)}{P(T', I)}$? Yes, with simple rules of probability, we find that it is the same as the conditional probability ratio that we can easily get from the CLM head:

$\frac{P(T | I)}{P(T' | I)} = \frac{\frac{P(T, I)}{P(I)}}{\frac{P(T', I)}{P(I)}} = \frac{P(T, I)}{P(T', I)}$

Combining the findings from Experiments 1 and 2, we get strong performance too:

| Model                                       | Image Score  |
|-------------------------------------------- | ------------ |
| Random Chance                               | 25.00        |
| BLIP Contrastive Head                       | 16.00        |
| BLIP ITM Head                               | 24.25        |
| DistilRoBERTa + BLIP CLM Head score ratios                                | **50.25**    |
| PaLI 17B ([with best known finetuning / prompting approach for Winoground](https://arxiv.org/abs/2305.10400)) | 41.50    |
| VQ2 ([with best known finetuning / prompting approach for Winoground](https://arxiv.org/abs/2305.10400)) | 42.20 |

## Conclusion

We've provided two approaches that beat the current state of the art for the Winoground image score, and one of them does not require negative training examples at all.

Will this approach scale to real-world retrieval? It is unclear - we need to go beyond these fun little experiments. The score ratios idea might only work well when we are retrieving from a set of images which already have the right objects in them, but possibly the wrong relationships between the objects. In this case, it might work well as a way to re-rank the top retrieved images from a database, but not as a full retrieval score by itself.
