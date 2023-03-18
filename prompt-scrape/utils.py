from datasets import load_dataset
import jsonlines
from tqdm import tqdm

potential_datasets = ['adversarial_qa_dbert_answer_the_following_q', 'adversarial_qa_dbert_based_on', 'adversarial_qa_dbert_generate_question', 'adversarial_qa_dbert_question_context_answer', 'adversarial_qa_dbert_tell_what_it_is', 'adversarial_qa_dbidaf_answer_the_following_q', 'adversarial_qa_dbidaf_based_on', 'adversarial_qa_dbidaf_generate_question', 'adversarial_qa_dbidaf_question_context_answer', 'adversarial_qa_dbidaf_tell_what_it_is', 'adversarial_qa_droberta_answer_the_following_q', 'adversarial_qa_droberta_based_on', 'adversarial_qa_droberta_generate_question', 'adversarial_qa_droberta_question_context_answer', 'adversarial_qa_droberta_tell_what_it_is', 'ag_news_classify', 'ag_news_classify_question_first', 'ag_news_classify_with_choices', 'ag_news_classify_with_choices_question_first', 'ag_news_recommend', 'ag_news_which_section', 'ag_news_which_section_choices']

for set in tqdm(potential_datasets):
    dataset = load_dataset("bigscience/P3", f"{set}")

    # print(dataset['train']['inputs_pretokenized'][0])

    with jsonlines.open('all_data.jsonl', mode='a') as writer:
        for datum in tqdm(dataset["train"]["inputs_pretokenized"]):
            writer.write({"prompt": datum, "subset": set})