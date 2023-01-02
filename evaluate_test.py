#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import random
from argparse import ArgumentParser
from pprint import pformat
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid, Softmax, CrossEntropyLoss
from data_utils import get_testdata_loaders, add_special_tokens_
from inference_test import sample_sequence
from ignite.metrics import Accuracy
from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
logger = logging.getLogger(__file__)
from setproctitle import setproctitle
setproctitle('Yoonna')

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]

# TODO : Augment with https://en.wikipedia.org/wiki/Most_common_words_in_English
# Rough version allowing knowledge duplicates

def run():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="data/new_valid_json_v3.json", help="Path or url of the dataset. If empty download from S3.") #data
    parser.add_argument("--test_dataset_cache", type=str, default='data/new_focus_cache_v5_ner_wo_rel.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="", help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--select_max_len", action='store_true', help="Select maximum length beam")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--seed", type=int, default=19950604, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--literal", type=float, default=1.0, help="Constraining based on literal")
    parser.add_argument("--graph", type=float, default=1.0, help="Constraining based on graph")
    parser.add_argument("--ner", type=float, default=1.0, help="Constraining based on NER")
    parser.add_argument("--eos_std_adj", type=float, default=0.0, help="Equation to adjust EOS token probability")
    parser.add_argument("--markov_graph", type=float, default=1.0, help="Constraining based on markov characteristics of graph")
    parser.add_argument("--markov_graph_v2", type=float, default=1.0, help="Constraining based on graph subject utilizing all relation / objects")
    parser.add_argument("--constrain", action='store_true', help="New constraining mechanism")
    parser.add_argument("--gen_eval", action='store_true', help="Generation Auto Evaluation")
    parser.add_argument("--without_rel", action="store_true", help="Whether to remove relation from graph data")
    parser.add_argument("--submit_path", type=str, default="", help="Line-by-line output for submission")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size (currently only performs basic override)")
    parser.add_argument("--beam_length_alpha", type=float, default=0.0, help="Beam alpha discount")
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
    logger.info(f"Submit to : {args.submit_path}")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2PK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = BARTPK_ctxt.from_pretrained(args.model_checkpoint)
        logger.info("Model loaded")

        model.to(args.device)
        logger.info("To Device")
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-decoder':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2PK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-encdec':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = BARTPK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError

    #dataset = get_dataset_only_train(tokenizer, args.dataset_path, args.dataset_cache)
    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer, generation=True)

    with torch.no_grad():
        r1 = 0
        r2 = 0
        rl = 0
        bleu = 0
        bertscore_precision = 0
        bertscore_recall = 0
        bertscore_f1 = 0
        bleurt = 0
        mauve = 0
        chrf1 = 0
        rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bleu_metric = load_metric("sacrebleu")
        bertscore_metric = load_metric("bertscore")
        bleurt_metric = load_metric("bleurt")
        # mauve_metric = load_metric("mauve")
        chrf_metric = CHRFScore()

        pg = Accuracy()
        kg = Accuracy()

        pass_num = 0
        test_pass_list = []

        for test_data_index, test_data in enumerate(tqdm(test_loader)):
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # TODO : Update can_idx for knowledge and persona when testing
            if model.config.model_type == 'gpt2':
                input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = test_data

            elif model.config.model_type == 'bart':
                # import pdb; pdb.set_trace()
                input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog, filtered_triples, \
                ner_cans = test_data
            else:
                raise NotImplementedError

            mask = (reply != tokenizer.pad_token_id)
            reply = reply[mask]

            #print('reply.tolist() ', reply.tolist())
            if len(reply.tolist()) == 0:
                pass_num += 1
                test_pass_list.append(test_data_index)
                continue

            if model.config.model_type == 'gpt2':
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
                persona_tensor = torch.tensor([persona]).cuda(device)
                knowledge_tensor = torch.tensor([knowledge]).cuda(device)
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
                p_index_cvtd = persona_pred_sigmoid


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


                k_index_1 = k_index_1.squeeze(0)
                k_index_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=args.device)

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

                out_ids = sample_sequence(final_input_tensor, token_type_ids=final_input_tti_tensor,
                                          decoder_input_ids=None, tokenizer=tokenizer, model=model, args=args, current_output=None)

            elif model.config.model_type == 'bart':
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
                max_position = 1024

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

                p_index_cvtd = persona_pred_sigmoid

                softmax = Softmax(dim=-1)
                knowledge_softmax = softmax(knowledge_logits)
                _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                all_knowledge_pred = []
                all_k_graph_pred = []
                all_ner_pred = []
                # all_k_graph_raw_pred = []
                for batch_i in range(args.test_batch_size):
                    knowledge_pred_idx = k_index_1[batch_i]
                    knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
                    mask_knowledge = torch.ne(knowledge_pred, padding)
                    knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                    knowledge_pred = knowledge_pred[1:-2]
                    all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos

                    k_graph_pred = filtered_triples[batch_i][knowledge_pred_idx]
                    mask_k_graph = torch.ne(k_graph_pred, padding)
                    k_graph_pred = torch.masked_select(k_graph_pred, mask_k_graph)
                    all_k_graph_pred.append(k_graph_pred)

                    ner_pred = ner_cans[batch_i][knowledge_pred_idx]
                    mask_ner = torch.ne(ner_pred, padding)
                    ner_pred = torch.masked_select(ner_pred, mask_ner)
                    all_ner_pred.append(ner_pred)

                    # TODO : Any processing?
                    # all_k_graph_raw_pred.append([[tokenizer.decode(value) for value in triple] for triple in filtered_triples_raw[batch_i]])

                # import pdb; pdb.set_trace()

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
                    new_input_size = new_input.size()[0]

                    if new_input_size < int(max_position) :
                        padding_size = int(max_position) -new_input_size
                        add_padding = torch.tensor([padding]*padding_size).cuda(device)
                        final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_input_list.append(final_input)
                final_input_tensor = torch.stack(final_input_list)
                decoder_input_ids = bos_tensor.unsqueeze(0)
                # print("Knowledge used : " + tokenizer.decode(all_knowledge_pred[0]))
                # TODO :
                out_ids = sample_sequence(final_input_tensor, token_type_ids=None, decoder_input_ids=decoder_input_ids,
                                          tokenizer=tokenizer, model=model, args=args, current_output=None, ner_cand_ids=all_ner_pred,
                                          knowledge_cand_ids=all_knowledge_pred, k_graph_cand_ids=all_k_graph_pred)

            machine, human, persona, knowledge = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
            special_tokens_list = [machine, human, persona, knowledge, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]


            gold_reply = reply
            #print('reply', reply)

            for special_token in special_tokens_list:
                   out_ids = [value for value in out_ids if value != special_token]

            gold_reply = tokenizer.decode(gold_reply.tolist(), skip_special_tokens=True)
            pred_reply = tokenizer.decode(out_ids, skip_special_tokens=True)
            print('gold: ', gold_reply)
            print('pred: ', pred_reply)
            print('---------------------------------------')

            submit_path = args.submit_path
            if submit_path:
                # TODO : Check speed?
                with open(submit_path, 'a') as file:
                    file.write(f"{pred_reply}\n")

            #ROUGE
            r = rouge_metric.score(pred_reply, gold_reply)
            r1 += r['rouge1'].fmeasure
            r2 += r['rouge2'].fmeasure
            rl += r['rougeL'].fmeasure

            #BLEU1,2,3,4 / BLEU avg
            bleu += bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']

            if args.gen_eval:
                # print(pred_reply)
                # print(gold_reply)

                #BERTScore
                bertscore_dict = bertscore_metric.compute(predictions=[pred_reply], references=[gold_reply], lang="en")
                bertscore_f1 += sum(bertscore_dict['f1'])
                bertscore_precision += sum(bertscore_dict['precision'])
                bertscore_recall += sum(bertscore_dict['recall'])

                #BLEURT
                bleurt += sum(bleurt_metric.compute(predictions=[pred_reply], references=[gold_reply])["scores"])

                #MAUVE
                # mauve += mauve_metric.compute(predictions=[pred_reply], references=[gold_reply]).mauve

            #CharF1
            chrf1 += chrf_metric([pred_reply], [[gold_reply]])

            # TODO : Remove below
            # PG
            p_label_cvtd = torch.tensor([1 if num in persona_grounding else 0 for num in range(5)], device=args.device)
            pg.update((p_index_cvtd.squeeze(), p_label_cvtd))
            pg_res = pg.compute()

            # KG
            k_label_cvtd = torch.tensor([1 if num in knowledge_grounding else 0 for num in range(10)], device=args.device)
            kg.update((knowledge_softmax, k_label_cvtd))
            kg_res = kg.compute()


        chrf1_result = chrf1/(test_data_index+1-pass_num)
        rouge1_result = r1/(test_data_index+1-pass_num)
        rouge2_result = r2/(test_data_index+1-pass_num)
        rougel_result = rl/(test_data_index+1-pass_num)
        bleu_result = bleu/(test_data_index+1-pass_num)
        bertscore_f1_result = bertscore_f1/(test_data_index+1-pass_num)
        bertscore_precision_result = bertscore_precision/(test_data_index+1-pass_num)
        bertscore_recall_result = bertscore_recall/(test_data_index+1-pass_num)
        bleurt_result = bleurt/(test_data_index+1-pass_num)
        # mauve_result = mauve/(test_data_index+1-pass_num)
        pg_result = pg_res
        kg_result = kg_res
        print('test pass list: ', test_pass_list, 'len: ', pass_num)
        print("F1: ", chrf1_result)
        print("ROUGE1", rouge1_result)
        print("ROUGE2", rouge2_result)
        print("ROUGEL", rougel_result)
        print("avg BLEU: ", bleu_result)
        if args.gen_eval:
            print("BERTScore F1: ", bertscore_f1_result)
            print("BERTScore Precision: ", bertscore_precision_result)
            print("BERTScore Recall: ", bertscore_recall_result)
            print("BLEURT", bleurt_result)
            # print("MAUVE", mauve_result)
        print("PG: ", pg_result)
        print("KG: ", kg_result)


if __name__ == "__main__":
    run()
