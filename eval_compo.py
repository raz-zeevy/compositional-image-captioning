"""Evaluate an image captioning model on the specified evaluation set using the specified set of evaluation metrics"""

import logging

import argparse
import sys

import pandas
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from typing import List, Dict

from datasets import *
from metrics import recall_pairs, beam_occurrences
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import numpy as np

from utils import (
    get_caption_without_special_tokens,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
    IMAGES_FILENAME,
    BOTTOM_UP_FEATURES_FILENAME,
    MODEL_SHOW_ATTEND_TELL,
    MODEL_BOTTOM_UP_TOP_DOWN,
    get_eval_log_file_path,
    decode_caption,
    TOKEN_PADDING,
    MODEL_BOTTOM_UP_TOP_DOWN_RANKING,
)

# from analysis_utils.visualize_attention import visualize_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size

METRIC_BLEU = "bleu"
METRIC_RECALL = "recall"
METRIC_BEAM_OCCURRENCES = "beam-occurrences"


def get_top_ranked_captions_indices(embedded_image, embedded_captions):
    # Compute similarity of image to all captions
    d = np.dot(embedded_image, embedded_captions.T).flatten()
    inds = np.argsort(d)[::-1]
    return inds


def re_rank_beam(
        decoder,
        top_k_generated_captions,
        encoded_features,
        word_map,
        coco_id,
        print_captions,
):
    """
    Re-rank the top-k generated captions using an image captioning model.

    Args:
        decoder (torch.nn.Module): The image captioning decoder model.
        top_k_generated_captions (list): List of top-k generated captions.
        encoded_features (torch.Tensor): Encoded features of the image.
        word_map (dict): Mapping of words to indices.
        coco_id (int): COCO ID of the image.
        print_captions (bool): Flag indicating whether to print the captions.

    Returns:
        list: Re-ranked top-k generated captions.
    """
    if print_captions:
        logging.info("COCO ID: {}".format(coco_id))
        logging.info("Before re-ranking:")
        for caption in top_k_generated_captions[:5]:
            logging.info(
                " ".join(
                    decode_caption(
                        get_caption_without_special_tokens(caption, word_map),
                        word_map
                    )
                )
            )

    lengths = [len(caption) - 1 for caption in top_k_generated_captions]
    top_k_generated_captions = torch.tensor(
        [
            top_k_generated_caption
            + [word_map[TOKEN_PADDING]]
            * (max(lengths) + 1 - len(top_k_generated_caption))
            for top_k_generated_caption in top_k_generated_captions
        ],
        device=device,
    )
    image_embedded, image_captions_embedded = decoder.forward_ranking(
        encoded_features, top_k_generated_captions,
        torch.tensor(lengths, device=device)
    )
    image_embedded = image_embedded.detach().cpu().numpy()[0]
    image_captions_embedded = image_captions_embedded.detach().cpu().numpy()

    indices = get_top_ranked_captions_indices(image_embedded,
                                              image_captions_embedded)
    top_k_generated_captions = [top_k_generated_captions[i] for i in indices]

    return [caption.cpu().numpy() for caption in top_k_generated_captions]


def evaluate(
        split_dataset_path,
        coco_val_folder,
        annotations_path,
        checkpoint_path,
        metrics,
        beam_size,
        eval_beam_size,
        re_ranking,
        nucleus_sampling,
        visualize,
        print_beam,
        print_captions,
):
    logging.basicConfig(level=logging.INFO)
    model_name = os.path.basename(checkpoint_path).split(".")[0]
    logging.info("Model: {}".format(model_name))

    # Load test_ids
    test_ids = []
    with open(split_dataset_path, "r") as file:
        test_ids = json.load(file)['test_images_split']
        test_ids = list(map(lambda x: int(x), test_ids))
    #load captions:
    captions_series = get_captions(test_ids, annotations_path)
    # create Predictor
    from clip_utils.predict import Predictor
    predictor = Predictor()
    predictor.setup(checkpoint_path)

    use_beam_search = True
    # Lists for target captions and generated captions for each imag
    target_captions = {}
    generated_captions = {}
    generated_beams = {}

    # Iterate over all .jpg files in the data_folder and predict captions
    for coco_id in tqdm(test_ids,
                        desc=f"PREDICTING CAPTIONS FOR TEST_IMAGES"):
        img_name = "COCO_val2014_" + str(coco_id).zfill(12) + ".jpg"
        img_path = os.path.join(coco_val_folder, img_name)
        top_k_generated_captions = predictor.predict(img_path, 'coco',
                                            use_beam_search=use_beam_search)
        generated_captions[coco_id] = top_k_generated_captions[0]
        target_captions[coco_id] = captions_series.loc[coco_id].tolist()
        store_beam = True if METRIC_BEAM_OCCURRENCES in metrics else False
        store_beam = store_beam and use_beam_search
        if store_beam:
            generated_beams[coco_id] = top_k_generated_captions

        # if nucleus_sampling:
        #     top_k_generated_captions, alphas, beam = decoder.nucleus_sampling(
        #         encoded_features,
        #         beam_size,
        #         top_p=nucleus_sampling,
        #         print_beam=print_beam,
        #     )
        # if store_beam:
        #     top_k_generated_captions, alphas, beam = decoder.beam_search(
        #         encoded_features,
        #         beam_size,
        #         store_alphas=visualize,
        #         store_beam=store_beam,
        #         print_beam=print_beam,
        #     )

        # if visualize:
        #     logging.info("Image COCO ID: {}".format(coco_id))
        #     for caption, alpha in zip(top_k_generated_captions, alphas):
        #         visualize_attention(
        #             image_features.squeeze(0), caption, alpha, word_map, smoothen=True
        #         )

        # if re_ranking:
        #     top_k_generated_captions = re_rank_beam(
        #         decoder,
        #         top_k_generated_captions,
        #         encoded_features,
        #         word_map,
        #         coco_id,
        #         print_captions,
        #     )

        # Change the size of the beam to eval_beam_size
        # old version
        # generated_captions[coco_id] = top_k_generated_captions[:eval_beam_size]
        if store_beam:
            generated_beams[coco_id] = generated_beams[coco_id][
                                          :eval_beam_size]

        if print_captions:
            logging.info("COCO ID: {}".format(coco_id))
            logging.info(generated_captions[coco_id])
            # for caption in generated_captions[coco_id]:
            #     logging.info(
            #         "\n".join(
            #             decode_caption(
            #                 get_caption_without_special_tokens(caption, word_map),
            #                 word_map,
            #             )
            #         )
            #     )


    assert len(target_captions) == len(generated_captions)

    # Save results
    name = str(os.path.basename(checkpoint_path).split(".")[0])
    if re_ranking:
        name += "_re_ranking"
    if nucleus_sampling:
        name += "_nucleus_sampling_p_" + str(nucleus_sampling)
    results_output_file_name = "results_" + name + ".json"

    results = []
    for coco_id, caption in generated_captions.items():
        results.append({"image_id": int(coco_id), "caption": caption})
    json.dump(results, open(results_output_file_name, "w"))

    # Calculate metric scores
    # eval_output_file_name = "eval_" + name + ".json"
    for metric in metrics:
        calculate_metric(
            metric,
            target_captions,
            generated_captions,
            generated_beams,
            # word_map,
            # dataset_splits_dict["heldout_pairs"],
            beam_size,
            # eval_output_file_name,
        )


def calculate_metric(
        metric_name,
        target_captions : Dict[int, List[str]],
        generated_captions : Dict[int, str],
        generated_beams : Dict[int, List[str]],
        # word_map,
        # heldout_pairs,
        beam_size,
        # output_file_name,
):
    if metric_name == METRIC_BLEU:
        '''
        formats:
        target_captions : [[ref1,ref2]] reg = ["word_1", "word_2",...]
        generated_captions = [hypo]
        '''
        generated_captions = [
            caption.split(" ")
            for caption in generated_captions.values()
        ]
        target_captions = [
            [ref.split(" ") for ref in caption]
            for caption in target_captions.values()
        ]
        bleu_1 = corpus_bleu(target_captions, generated_captions,
                             weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(
            target_captions, generated_captions, weights=(0.5, 0.5, 0, 0)
        )
        bleu_3 = corpus_bleu(
            target_captions, generated_captions, weights=(0.33, 0.33, 0.33, 0)
        )
        bleu_4 = corpus_bleu(
            target_captions, generated_captions,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_scores = [bleu_1, bleu_2, bleu_3, bleu_4]
        bleu_scores = [float("%.2f" % elem) for elem in bleu_scores]
        logging.info(
            "\nBLEU score @ beam size {} is {}".format(beam_size, bleu_scores))
    # elif metric_name == METRIC_RECALL:
    #     recall_pairs(generated_captions, word_map, heldout_pairs,
    #                  output_file_name)
    # elif metric_name == METRIC_BEAM_OCCURRENCES:
    #     beam_occurrences_score = beam_occurrences(
    #         generated_beams, beam_size, word_map, heldout_pairs
    #     )
    #     logging.info(
    #         "\nBeam occurrences score @ beam size {} is {}".format(
    #             beam_size, beam_occurrences_score
    #         )
    #     )


def get_captions(test_ids: list, annotations_path : str) -> pandas.Series:
    import pandas as pd
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    captions = pd.DataFrame(data['annotations']).set_index('image_id')
    captions = captions.loc[test_ids]
    return captions["caption"]


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_val_folder",
        help="Folder where the COCO validation images are located",
        default=os.path.expanduser("datasets/val2014/"),
        # TODO: required=True
    )
    parser.add_argument(
        "--annotations_path",
        help="Path to the json file containing the annotations",
        default=os.path.expanduser("datasets/annotations/captions_val2014.json"),
        # TODO: required=True
    )
    parser.add_argument(
        "--split_dataset_path", help="Json file with the split id's",
        default=os.path.expanduser(
            "data/dataset_splits/dataset_splits_1.json"),
        # TODO: required=True
    )
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model",
        default=os.path.expanduser(
            "checkpoint/coco_prefix_latest.pt"),
        # TODO: required=True
    )
    parser.add_argument(
        "--metrics",
        help="Evaluation metrics",
        nargs="+",
        default=[METRIC_BLEU],
        choices=[METRIC_BLEU, METRIC_RECALL, METRIC_BEAM_OCCURRENCES],
    )

    parser.add_argument(
        "--beam-size", help="Size of the decoding beam", type=int, default=5
    )
    parser.add_argument(
        "--eval-beam-size",
        help="Number of sequences from the beam that should be used for evaluation",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--re-ranking",
        help="Use re-ranking to sort the beam",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--nucleus-sampling",
        help="Use nucleus sampling with the given p instead of beam search",
        type=float,
    )
    parser.add_argument(
        "--visualize-attention",
        help="Visualize the attention for every sample",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--print-beam",
        help="Print the decoding beam for every sample",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--print-captions",
        help="Print the generated captions for every sample",
        default=False,
        action="store_true",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    # logging.basicConfig(
    #     filename=get_eval_log_file_path(
    #         parsed_args.checkpoint, parsed_args.dataset_splits
    #     ),
    #     level=logging.INFO,
    # )
    logging.info(parsed_args)
    evaluate(
        coco_val_folder=parsed_args.coco_val_folder,
        annotations_path=parsed_args.annotations_path,
        split_dataset_path=parsed_args.split_dataset_path,
        checkpoint_path=parsed_args.checkpoint,
        metrics=parsed_args.metrics,
        beam_size=parsed_args.beam_size,
        eval_beam_size=parsed_args.eval_beam_size,
        re_ranking=parsed_args.re_ranking,
        nucleus_sampling=parsed_args.nucleus_sampling,
        visualize=parsed_args.visualize_attention,
        print_beam=parsed_args.print_beam,
        print_captions=parsed_args.print_captions,
    )
    # hypo = "Almost everything we want can be achieved through kosher"
    # ref = "Almost everything we want can be achieved through kosher"
    # target_captions = {1 : [ref]}
    # generated_captions = {1 : hypo}
    # generated_captions = [
    #     caption.split(" ")
    #     for caption in generated_captions.values()
    # ]
    # target_captions = [
    #     [ref.split(" ") for ref in caption]
    #     for caption in target_captions.values()
    # ]
    # bleu_1 = corpus_bleu(target_captions, generated_captions,
    #                      weights=[1])
    # print(bleu_1)
