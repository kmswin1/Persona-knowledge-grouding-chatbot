#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

import logging
import random
import json
from argparse import ArgumentParser
from pprint import pformat
import warnings
from torch.nn import Sigmoid, Softmax
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from data_utils import get_testdata_loaders, add_special_tokens_
from node_graph.node import MarkovNode

logger = logging.getLogger(__file__)

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]

# TODO : Augment with https://en.wikipedia.org/wiki/Most_common_words_in_English
# Rough version allowing knowledge duplicates
NONE_TOKENS = [('<pad>', 3916773), ('.', 106376), (',', 98585), ('<s>', 84585), ('</s>', 84585),
               ('<knowledge>', 56390), ('<persona>', 28195)]

SKIP_TOKENS = [('<pad>', 3916773), ('Ġthe', 140818), ('.', 106376), (',', 98585), ('<s>', 84585), ('</s>', 84585),
               ('Ġof', 71110), ('<knowledge>', 56390), ('Ġand', 53218), ('Ġin', 46310), ('Ġto', 45793), ('Ġa', 37956),
               ('I', 28263), ('<persona>', 28195), ('Ġwas', 22661), ('Ġis', 19035), ('The', 17900), ('-', 16593),
               ('Ġby', 14166), ('Ġon', 13663), ('Ġfor', 13092), ('Ġ(', 13024), ('Ġas', 12949), ('Ġwith', 10944),
               ('Ġfrom', 10269), ("'s", 10088), ('Ġat', 9384), ('Ġlike', 9356), ('Ġthat', 8589), ('ĠThe', 8554),
               (')', 7778), ('Ġwere', 7434), ('Ġhave', 6944), ('Ġan', 6856), ('Ġwhich', 6437), ('Ġare', 6371),
               ('Ġam', 6336), ('Ġit', 6193), ('Âł', 5865), ('Ġits', 5110), ('Ġbeen', 4901), ('Ġwould', 4865),
               ('Ġ"', 4776), ('s', 4582), ('Ġhas', 4578), ('In', 4330), ('Ġbe', 4315), ('Ġalso', 3916),
               ('Ġhad', 3901)]

def top_filtering(logits, tokenizer, top_k=0., top_p=0.9, constrain_ids=None, markov_constrain_ids=None, threshold=-float('Inf'),
                  filter_value=-float('Inf'), markov_idx=0, markov_graph_nodes=None):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    # import pdb; pdb.set_trace()
    if constrain_ids is not None:
        for id, mult in constrain_ids.items():
            # Multiply probability in relation to other logits, let's do this at last
            logits[id] = logits[id] + np.log(mult)
    if markov_constrain_ids is not None:
        if markov_idx < len(markov_constrain_ids):
            id, mult = markov_constrain_ids[markov_idx]
            # FIXME : THIS SHOULD BE SOFT REPR BASED MATCHING
            logits[id] = logits[id] + np.log(mult)
    if markov_graph_nodes is not None and len(markov_graph_nodes) > 0:
        for id in logits:
            id_repr = tokenizer.decode([id]).lower().strip()
            if id_repr in markov_graph_nodes:
                logits[id] = logits[id] + np.log(max({node.mult for node in markov_graph_nodes[id_repr]}))

    return logits


def sample_sequence(input_ids, token_type_ids, decoder_input_ids, tokenizer, model, args, current_output=None,
                    ner_cand_ids=None, knowledge_cand_ids=None, k_graph_cand_ids=None,
                    # This is only for certain markov constraints
                    k_graph_multiple_markov_triple_ids=None
                    ):
    beam_size = args.beam_size
    beam_search = beam_size > 1
    beam_init_step = False

    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    machine = [special_tokens_ids[0]]
    #print('machine: ', machine)
    #print('input_ids', input_ids.size(), 'tti', token_type_ids.size())

    gpt_input_ids = input_ids[0]
    if token_type_ids is not None:
        gpt_tti = token_type_ids[0]
    if decoder_input_ids is not None:
        bart_decoder_first_token = decoder_input_ids[0]
    if current_output is None:
        if beam_search:
            # FIXME : Early completion support
            # comp_beams = set()
            current_output = [[] for _ in range(beam_size)]
            cur_beam_logprobs=torch.zeros((1, 1)).cuda() # 1d : hypothesis x log score sum, start with 1 x 1
            beam_init_step = True
        else:
            current_output = []

    # import pdb; pdb.set_trace()

    k_graph_cand_ids[0] = [tok for tok in k_graph_cand_ids[0] if tok != -100]
    ner_cand_ids[0] = [tok for tok in ner_cand_ids[0] if tok != -100]
    # import pdb; pdb.set_trace()
    try:
        print(f"NER : {tokenizer.decode(ner_cand_ids[0], skip_special_tokens=True)}")
        print(f"Knowledge : {tokenizer.decode(knowledge_cand_ids[0], skip_special_tokens=True)}")
        print(f"Graph : {tokenizer.decode(k_graph_cand_ids[0], skip_special_tokens=True)}")
    except Exception as e:
        print(f"Error with log; skipping")

    none_token_ids = tokenizer.convert_tokens_to_ids([tok[0] for tok in NONE_TOKENS])
    skip_token_ids = tokenizer.convert_tokens_to_ids([tok[0] for tok in SKIP_TOKENS])

    constrain_ids = {}
    if args.constrain:
        # TODO : Support constraint lower than 1.0
        if args.graph != 1.0:
            for id in k_graph_cand_ids[0]:
                if id in skip_token_ids:
                    continue
                prev_constrain = constrain_ids.get(id)
                constrain_ids[id] = args.graph if not prev_constrain else max(prev_constrain, args.graph)
        if args.ner != 1.0:
            for id in ner_cand_ids[0]:
                if id in skip_token_ids:
                    continue
                prev_constrain = constrain_ids.get(id)
                constrain_ids[id] = args.ner if not prev_constrain else max(prev_constrain, args.ner)
        if args.literal != 1.0:
            for id in knowledge_cand_ids[0]:
                if id in skip_token_ids:
                    continue
                prev_constrain = constrain_ids.get(id)
                constrain_ids[id] = args.literal if not prev_constrain else max(prev_constrain, args.literal)

    # Only supports single triple
    markov_constrain_ids = []
    # TODO : Different multiplication for markov constraint
    if args.markov_graph != 1.0:
        for id in k_graph_cand_ids[0]:
            if id in none_token_ids:
                continue
            markov_constrain_ids.append((id, args.markov_graph))

    # Dictionary of nodes keyed on string representation
    import collections
    markov_graphs_nodes = collections.defaultdict(set)
    if args.markov_graph_v2 != 1.0:
        for triple in k_graph_multiple_markov_triple_ids:
            subj_ids, rel_ids, obj_ids = triple
            root_node = None
            subj_node = None
            for subj_id in subj_ids:
                if subj_node is None:
                    root_node = MarkovNode(subj_id, tokenizer.decode([subj_id]).lower().strip(), args.markov_graph_v2)
                    subj_node = root_node
                else:
                    new_node = MarkovNode(subj_id, tokenizer.decode([subj_id]).lower().strip(), args.markov_graph_v2)
                    subj_node.children.add(new_node)
                    subj_node = new_node

            # TODO : Refactor
            rel_node = None
            for rel_id in rel_ids:
                if rel_node is None:
                    rel_node = MarkovNode(rel_id, tokenizer.decode([rel_id]).lower().strip(), args.markov_graph_v2)
                    root_node.children.add(rel_node)
                else:
                    new_node = MarkovNode(rel_id, tokenizer.decode([rel_id]).lower().strip(), args.markov_graph_v2)
                    rel_node.children.add(new_node)
                    rel_node = new_node

            obj_node = None
            for obj_id in obj_ids:
                if obj_node is None:
                    obj_node = MarkovNode(obj_id, tokenizer.decode([obj_id]).lower().strip(), args.markov_graph_v2)
                    root_node.children.add(obj_node)
                else:
                    new_node = MarkovNode(obj_id, tokenizer.decode([obj_id]).lower().strip(), args.markov_graph_v2)
                    obj_node.children.add(new_node)
                    obj_node = new_node

            markov_graphs_nodes[root_node.repr].add(root_node)

    # print(f"CONSTRAIN_TOKENS = {constrain_ids}")
    print(f"CONSTRAIN_TOKENS_LITERAL = {[(tokenizer.decode([key]), value) for key, value in constrain_ids]}")
    # print(f"MARKOV CONSTRAIN_TOKENS = {markov_constrain_ids}")
    print(
        f"MARKOV CONSTRAIN_TOKENS_LITERAL = {[(tokenizer.decode([key]), value) for key, value in markov_constrain_ids]}")

    markov_graph_idx = 0

    # import pdb; pdb.set_trace()
    for ind in range(args.max_length):
        if beam_search:
            beam_logits = []
            assert model.config.model_type == 'bart'
            # Beam Initialization

            if beam_init_step:
                # Only initiating 1 copy of the logits
                if len(decoder_input_ids) > 0 and decoder_input_ids[-1] == 2:
                    # Reusing previous logits
                    logits = torch.zeros(logits.size)
                    logits[2] = 1.0
                else:
                    output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                    logits = output[0]
                beam_logits.append(logits)
                beam_init_step = False
            else:
                # TODO : 2D input
                # TODO : only using non-completed beams
                for i, beam in enumerate(current_output):
                    if len(beam) > 0:
                        decoder_input_ids = torch.cat([bart_decoder_first_token, torch.tensor(beam).type_as(input_ids)], dim=-1).unsqueeze(0)

                    #print("input: ", input_ids, "dec: ", decoder_input_ids)
                    # MODEL HAPPENS HERE
                    output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                    logits = output[0]
                    beam_logits.append(logits)

        else:
            if model.config.model_type == 'gpt2':
                if len(current_output) > 0:
                    input_ids = torch.cat([gpt_input_ids, torch.tensor(current_output).type_as(input_ids)], dim=-1).unsqueeze(0)
                    token_type_ids = torch.cat([gpt_tti, torch.tensor(machine*len(current_output)).type_as(token_type_ids)]).unsqueeze(0)

                output = model(input_ids=input_ids, token_type_ids=token_type_ids)
                labels, logits = output[0], output[1]
                # TODO
                #print('logits', logits)
            elif model.config.model_type == 'bart':
                if len(current_output) > 0:
                    decoder_input_ids = torch.cat([bart_decoder_first_token, torch.tensor(current_output).type_as(input_ids)], dim=-1).unsqueeze(0)

                #print("input: ", input_ids, "dec: ", decoder_input_ids)
                # MODEL HAPPENS HERE
                output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                logits = output[0]
            else:
                raise NotImplementedError

        # import pdb; pdb.set_trace()

        if beam_search:
            if not beam_logits:
                break
            # TODO : LENGTH DEPRECATIONS & MARKOVS
            beam_logits = [logits[0, -1, :] / args.temperature for logits in beam_logits]
            # skipping top_filtering for now
            # beam_size x vocab_size
            # import pdb; pdb.set_trace()
            beam_logprobs = [F.log_softmax(logits, dim=-1) for logits in beam_logits]

            # Min length application
            if ind < args.min_length:
                for logprobs in beam_logprobs:
                    logprobs[2] = -1e20     # Negative Infinity for EOS token

            # beam_size x vocab_size
            # import pdb; pdb.set_trace()
            per_token_logprobs = torch.add(
                cur_beam_logprobs if len(cur_beam_logprobs) == 1 else cur_beam_logprobs.unsqueeze(1),
                torch.stack(beam_logprobs))
            # import pdb; pdb.set_trace()
            # Only place beam_size is relevant
            vec, idx = torch.topk(per_token_logprobs.flatten(), beam_size)
            idx = idx.cpu()
            # import pdb; pdb.set_trace()
            # Reformat to 2d array
            indices = np.array(np.unravel_index(idx.numpy(), per_token_logprobs.shape)).T

            # import pdb; pdb.set_trace()
            new_output = []
            new_logprobs = []
            for i, (beam_idx, token_idx) in enumerate(indices):
                # beam_idx changes everytime
                # Copy needed to append new token
                cand = current_output[beam_idx].copy()
                cand.append(token_idx)
                # Shuffling the beams
                new_output.append(cand)
                new_logprobs.append(per_token_logprobs[beam_idx][token_idx])

                # TODO : EOS detection and beam reuse
                # if token_idx in special_tokens_ids:
                #     comp_beams.add(i)
                #     # EOS
                # elif token_idx == 2:
                #     comp_beams.add(i)

            # import pdb; pdb.set_trace()
            current_output = new_output
            cur_beam_logprobs = torch.tensor(new_logprobs).cuda()

            # if len(comp_beams) >= beam_size:
            #     # All beams completed. Break
            #     print("Beam search completed")
            #     break

        else:
            logits = logits[0, -1, :] / args.temperature #size [50262]

            logits = top_filtering(logits, tokenizer, top_k=args.top_k, top_p=args.top_p, constrain_ids=constrain_ids,
                                   markov_constrain_ids=markov_constrain_ids, markov_idx=markov_graph_idx,
                                   markov_graph_nodes=markov_graphs_nodes)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

            if ind < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break

            repr_key = tokenizer.decode([prev.item()]).lower().strip()
            # TODO : Paraphrase tokening - stemming?
            if markov_graph_idx < len(markov_constrain_ids) and \
                    repr_key == tokenizer.decode([markov_constrain_ids[markov_graph_idx][0]]).lower().strip():
                print(f"Markov found : {markov_graph_idx}")
                markov_graph_idx += 1

            if repr_key in markov_graphs_nodes:
                print(f"Graph Markov Found : {repr_key}")
                for elem in markov_graphs_nodes[repr_key]:
                    markov_graphs_nodes[elem.repr].add(elem)
                del markov_graphs_nodes[repr_key]

            current_output.append(prev.item())

    # import pdb; pdb.set_trace()
    if beam_search:
        # Final beam selection

        for i, beam in enumerate(current_output):
            truncated_beam = []
            for j, tok in enumerate(beam):
                if tok == 2:
                    break
                truncated_beam.append(tok)
            beam_length = len(truncated_beam)

            lp = (5 + beam_length) ** args.beam_length_alpha / (5 + 1) ** args.beam_length_alpha
            cur_beam_logprobs[i] /= lp
            beam_logprob = cur_beam_logprobs[i]

            print(f"Beam {i} : {tokenizer.decode(truncated_beam)}, length : {beam_length}, "
                  f"norm_log_prob : {beam_logprob}, norm_prob : {torch.exp(beam_logprob)}")

        max_idx = torch.argmax(cur_beam_logprobs)
        if not args.select_max_len:
            final_output = current_output[max_idx]
            log_prob = cur_beam_logprobs[max_idx]
        else:
            prob_max_output = current_output[max_idx]
            max_len = len(prob_max_output)
            for i, tok_idx in enumerate(prob_max_output):
                if tok_idx == 2:
                    max_len = i
                    break

            res_idx = max_idx
            for i, beam in enumerate(current_output):
                length = len(beam)
                for j, tok_idx in enumerate(beam):
                    if tok_idx == 2:
                        length = j
                        break
                if length > max_len:
                    max_len = length
                    res_idx = i
            final_output = current_output[res_idx]
            log_prob = cur_beam_logprobs[res_idx]
            if res_idx != max_idx:
                print(f"Longer Beam {res_idx} selected.")

        print(f"Beam selected with index : {max_idx}, log_prob : {log_prob}, prob : {torch.exp(log_prob)}")
        print(f"All beam candidates :")
    else:
        final_output = current_output
    # import pdb; pdb.set_trace()

    return final_output


def run():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="data/valid_focus.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='data/focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="", help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--filename", type=str, default="", help="File name for saving output")
    parser.add_argument("--constrain", action='store_true', help="New constraining mechanism")
    parser.add_argument("--constrain_fixed_mult", type=float, default=2.0, help="Fixed constrain multiple")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    args.distributed = (args.local_rank != -1)

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Get model and tokenizer")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt as model
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = model.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt as model
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = model.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-decoder':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt as model
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = model.from_pretrained(args.model_checkpont)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-encdec':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt as model
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = model.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError


    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer)

    with open(args.test_dataset_path, 'r') as original_file:
        file = json.load(original_file)
        data = file['data']

    if args.filename is None:
        raise Exception('Please specify file name to save the generated outputs.')

    with open(args.filename + '.json', 'w') as outputfile:
        with torch.no_grad():
            alldict = dict()
            alllist = list()
            for data_index, test_data in enumerate(test_loader):
                print(data_index)
                outputdict = dict()
                if model.config.model_type == 'gpt2':
                    input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                    knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = test_data
                    output = model(
                        input_ids=input_ids,
                        input_eos=input_eos,
                        token_type_ids=token_type_ids,
                        only_dial_input_ids=dialog,
                        only_dial_token_type_ids=dialog_tti,
                        persona_input_ids=persona_candidates,
                        knowledge_input_ids=knowledge_candidates,
                        persona_can_idx=persona_can_idx,
                        knowledge_can_idx=knowledge_can_idx,
                        tot_knowledge=tot_knowledge,
                        tot_knowledge_token_ids=tot_knowledge_token_ids,
                        tot_knowledge_eos=tot_knowledge_eos,
                        training=False,
                        mc_token_ids=mc_token_ids
                    )
                    lm_labels, lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2], output[3]

                    machine, human, persona, knowledge, padding, bos = 50257, 50258, 50259, 50260, 50261, 50256

                    device = input_ids.get_device()

                    machine_tensor = torch.tensor([machine]).cuda(device)
                    human_tensor = torch.tensor([human]).cuda(device)
                    persona_tensor = torch.tensor([persona]).cuda(device)
                    knowledge_tensor = torch.tensor([knowledge]).cuda(device)
                    #padding_tensor = torch.tensor([padding]).cuda(device)
                    bos_tensor = torch.tensor([bos]).cuda(device)

                    sigmoid = Sigmoid()
                    persona_pred_sigmoid = sigmoid(persona_logits)
                    persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_candidates[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[:-2])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

                    softmax = Softmax(dim=-1)
                    knowledge_pred = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_pred, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(args.test_batch_size):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-2]
                        all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos

                    final_input_list = []
                    final_input_tti_list = []
                    for batch_i in range(args.test_batch_size):
                        only_dial_input_ids_batch = dialog[batch_i]
                        only_dial_token_type_ids_batch = dialog_tti[batch_i]
                        mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                        mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                        only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                        only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch, mask_only_dial_tti_batch)

                        if len(all_persona_pred[batch_i]) > 0:
                            concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                            new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                            new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).cuda(device)

                        else:
                            new_persona = None
                            new_persona_tti = None

                        new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                        new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).cuda(device)

                        only_dial_input_ids_batch = only_dial_input_ids_batch[1:-1]
                        only_dial_token_type_ids_batch = only_dial_token_type_ids_batch[1:]
                        if new_persona is not None:
                            new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, machine_tensor], dim=-1)
                            new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                        else:
                            new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, machine_tensor], dim=-1)
                            new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch], dim=-1)

                        final_input_list.append(new_input)
                        final_input_tti_list.append(new_input_tti)
                    final_input_tensor = torch.stack(final_input_list)
                    final_input_tti_tensor = torch.stack(final_input_tti_list)

                    out_ids = sample_sequence(final_input_tensor, token_type_ids=final_input_tti_tensor, decoder_input_ids=None, tokenizer=tokenizer, model=model, args=args, current_output=None)


                elif model.config.model_type == 'bart':
                    input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                    knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = test_data
                    #input_ids = input_ids.squeeze()
                    output = model(
                        input_ids=input_ids,
                        input_eos=input_eos,
                        only_dial_input_ids=dialog,
                        decoder_input_ids=decoder_input_ids,
                        persona_input_ids=persona_candidates,
                        knowledge_input_ids=knowledge_candidates,
                        persona_can_idx=persona_can_idx,
                        knowledge_can_idx=knowledge_can_idx,
                        tot_knowledge=tot_knowledge,
                        tot_knowledge_eos=tot_knowledge_eos,
                        training=False,
                        mc_token_ids=mc_token_ids
                    )
                    lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2]
                    persona, knowledge = 50267, 50268
                    bos, padding, eos = 0, 1, 2
                    device = input_ids.get_device()

                    persona_tensor = torch.tensor([persona]).cuda(device)
                    knowledge_tensor = torch.tensor([knowledge]).cuda(device)
                    bos_tensor = torch.tensor([bos]).cuda(device)
                    eos_tensor = torch.tensor([eos]).cuda(device)

                    sigmoid = Sigmoid()
                    persona_pred_sigmoid = sigmoid(persona_logits)
                    persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_candidates[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[:-2])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

                    softmax = Softmax(dim=-1)
                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(args.test_batch_size):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-2]
                        all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos

                    final_input_list = []
                    for batch_i in range(args.test_batch_size):
                        only_dial_input_ids_batch = dialog[batch_i]
                        mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                        only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                        if len(all_persona_pred[batch_i])>0:
                            concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                            new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                        else:
                            new_persona = None
                        new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                        if new_persona is not None:
                            new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                        else:
                            new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)
                        final_input_list.append(new_input)
                    final_input_tensor = torch.stack(final_input_list)
                    # Where RAG related change can go
                    decoder_input_ids = bos_tensor.unsqueeze(0)
                    # import pdb; pdb.set_trace()
                    print("Knowledge used : " + tokenizer.decode(all_knowledge_pred[0]))
                    out_ids = sample_sequence(final_input_tensor, token_type_ids=None,
                                              decoder_input_ids=decoder_input_ids, tokenizer=tokenizer, model=model,
                                              args=args, current_output=None, knowledge_cand_ids=all_knowledge_pred)


                mask = (reply != padding)
                reply = reply[mask]
                reply = tokenizer.decode(reply, skip_special_tokens=True)
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                print(data_index, out_text)
                outputdict['gold_answer'] = reply
                outputdict['pred_answer'] = out_text
                index = data[data_index//6]["dialogID"]
                outputdict['ID'] = index
                alllist.append(outputdict)

            alldict['data'] = alllist
        json.dump(alldict, outputfile)
        print("done!")

if __name__ == "__main__":
    run()
