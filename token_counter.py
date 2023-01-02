import logging
from pprint import pformat
from argparse import ArgumentParser

from data_utils import get_testdata_loaders, add_special_tokens_

logger = logging.getLogger(__file__)
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<machine>', '<human>', '<persona>', '<knowledge>']}

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]

def run():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="data/train_focus.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='data/focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="", help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=0, help="Top-K token frequencies to display")
    args = parser.parse_args()
    args.distributed = False
    args.test_batch_size = 1

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
    elif args.model_name == 'transformer-decoder':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    elif args.model_name == 'transformer-encdec':
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)

    orig_num_tokens = len(tokenizer.encoder)
    if type(tokenizer).__name__ == 'GPT2Tokenizer':
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = '<pad>'
        print('<pad> token added!')
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    print("orig num", orig_num_tokens, "num_added", num_added_tokens)  # 50265, 4

    import collections
    token_freqs = collections.Counter()
    token_map = {}

    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer)

    for data_index, test_data in enumerate(test_loader):
        input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, \
        persona_can_idx, persona_grounding, knowledge_candidates, knowledge_can_idx, knowledge_grounding, \
        tot_knowledge, tot_knowledge_eos, reply, dialog = test_data
        # import pdb; pdb.set_trace()
        in_ids = input_ids[0]
        for p in persona_candidates.squeeze():
            tokens = tokenizer.convert_ids_to_tokens(p)
            # assert len(in_ids) == len(tokens)
            for i, token in enumerate(tokens):
                token_freqs[token] += 1
                token_map[token] = in_ids[i]

        for k in knowledge_candidates.squeeze():
            tokens = tokenizer.convert_ids_to_tokens(k)
            # assert len(in_ids) == len(tokens)
            for i, token in enumerate(tokens):
                token_freqs[token] += 1
                token_map[token] = in_ids[i]
    print(token_freqs)
    print(f"Number of tokens : {len(token_freqs)}")
    commons = token_freqs.most_common(args.top_k)
    print(f"Most Common tokens : {commons}")
    print(f"Most Common token ids : {[token_map[token[0]] for token in commons]}")


if __name__ == "__main__":
    run()
