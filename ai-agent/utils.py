from stackapi import StackAPI
import config
import json
import ast
import asyncio
import datetime
from typing import List, Dict, Tuple
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import pinecone

from rank_bm25 import BM25Okapi

from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import random
import os
import logging

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_STACK_OVERFLOW_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_STACK_OVERFLOW_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_STACK_OVERFLOW_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
SO_CLIENT_SECRET = os.getenv('SO_CLIENT_SECRET')
SO_KEY = os.getenv('SO_KEY')

SITE = StackAPI('stackoverflow', client_secret=SO_CLIENT_SECRET, key=SO_KEY)

llm = ChatOpenAI(model="gpt-4", temperature=0.9)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, disallowed_special=(), model="text-embedding-3-small")

question_divider = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(config.QUESTION_DIVIDER_PROMPT)
)

keyword_extractor = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(config.KEYWORD_EXTRACTOR_PROMPT)
    )

question_complexity_checker = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(config.QUESTION_COMPLEXITY_CHECKER_PROMPT)
)

evidence_scorer = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(config.EVIDENCE_SCORER_PROMPT)
)

evidence_checker = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(config.EVIDENCE_CHECKER_PROMPT)
)

answer_generator = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(config.FINAL_ANSWER_GENERATOR_PROMPT)
)

global_evidence = ""
user_query = ""
global_unanswered_question_list = []
global_question_extracted_keywords_dict = {}
global_generated_answer = ""
global_question = ""


def choose_top2_unaccepted_answers(current_unaccepted_1: Dict, current_unaccepted_2: Dict, new_unaccepted: Dict) -> List[Dict]:
    """
    A function to choose top 2 unaccepted answers from stack-overflow from the provided 3 based on their scores and date
    :param current_unaccepted_1: One of the current top 2 answers
    :param current_unaccepted_2: One of the current top 2 answers
    :param new_unaccepted: new candidate answer
    :return: An updated list of top 2 answers
    """

    top2_list = []

    if current_unaccepted_1.get("score") > current_unaccepted_2.get("score") and new_unaccepted.get(
            "score") > current_unaccepted_2.get("score"):
        top2_list = [current_unaccepted_1, new_unaccepted]
    elif current_unaccepted_2.get("score") > current_unaccepted_1.get("score") and new_unaccepted.get(
            "score") > current_unaccepted_1.get("score"):
        top2_list = [current_unaccepted_2, new_unaccepted]
    elif current_unaccepted_1.get("score") > new_unaccepted.get("score") and current_unaccepted_2.get(
            "score") > new_unaccepted.get("score"):
        top2_list = [current_unaccepted_1, current_unaccepted_2]
    elif current_unaccepted_1.get("score") > new_unaccepted.get("score") and new_unaccepted.get(
            "score") == current_unaccepted_2.get("score"):
        top2_list.append(current_unaccepted_1)
        if datetime.strptime(new_unaccepted.get("creation_date"), "%Y-%m-%d %H:%M:%S") > datetime.strptime(
                current_unaccepted_2.get("creation_date"), "%Y-%m-%d %H:%M:%S"):
            top2_list.append(new_unaccepted)
        else:
            top2_list.append(current_unaccepted_2)
    elif current_unaccepted_2.get("score") > new_unaccepted.get("score") and new_unaccepted.get(
            "score") == current_unaccepted_1.get("score"):
        top2_list.append(current_unaccepted_2)
        if datetime.strptime(new_unaccepted.get("creation_date"), "%Y-%m-%d %H:%M:%S") > datetime.strptime(
                current_unaccepted_1.get("creation_date"), "%Y-%m-%d %H:%M:%S"):
            top2_list.append(new_unaccepted)
        else:
            top2_list.append(current_unaccepted_1)
    elif new_unaccepted.get("score") > current_unaccepted_1.get("score") and current_unaccepted_2.get(
            "score") == current_unaccepted_1.get("score"):
        top2_list.append(new_unaccepted)
        if datetime.strptime(current_unaccepted_1.get("creation_date"), "%Y-%m-%d %H:%M:%S") > datetime.strptime(
                current_unaccepted_2.get("creation_date"), "%Y-%m-%d %H:%M:%S"):
            top2_list.append(current_unaccepted_1)
        else:
            top2_list.append(current_unaccepted_2)
    elif current_unaccepted_2.get("score") == current_unaccepted_1.get("score") and current_unaccepted_2.get("score") == new_unaccepted.get("score"):
        top2_list = random.choices([current_unaccepted_1, current_unaccepted_2, new_unaccepted], k=2)

    top2_full_list = {"not_accepted_answer_body": [], "not_accepted_answer_date": []}
    for current_answer in top2_list:
        top2_full_list["not_accepted_answer_body"].append(current_answer.get("body"))
        top2_full_list["not_accepted_answer_date"].append(
            current_answer.get("creation_date")
        )
    return top2_full_list


def filter_and_process_answers(question_answer_list: List[Dict]) -> List[Dict]:
    """
    A function to filter and process answers from stack-overflow
    :param question_answer_list: A list of questions and their answers with their corresponding details (question owner, date, answer body, etc.)
    :return: A filtered and processed question-answer list with details, and unanswered questions with their links that might be useful
    """

    processed_answers = []

    for question_answer_pair in question_answer_list:
        current_processed_answer = {}

        question = question_answer_pair.get("question")
        answers = question_answer_pair.get("answers")
        top2_full_answers = []

        for answer in answers:
            current_processed_answer["question_title"] = question.get("title")
            current_processed_answer["question_link"] = question.get("link")
            if answer.get("is_accepted"):
                if answer.get("is_accepted") == True:
                    current_processed_answer["accepted_answer_body"] = answer.get("body")
                    current_processed_answer["accepted_answer_date"] = answer.get("creation_date")
            else:
                if not current_processed_answer.get("not_accepted_answer_body"):
                    current_processed_answer["not_accepted_answer_body"] = [answer.get("body")]
                    current_processed_answer["not_accepted_answer_date"] = [answer.get("creation_date")]
                    top2_full_answers.append(answer)

                else:
                    if len(current_processed_answer["not_accepted_answer_date"]) < 2:
                        current_processed_answer["not_accepted_answer_body"].append(answer.get("body"))
                        current_processed_answer["not_accepted_answer_date"].append(answer.get("creation_date"))
                        top2_full_answers.append(answer)

                    else:
                        top2_unaccepted = choose_top2_unaccepted_answers(top2_full_answers[0], top2_full_answers[1], answer)
                        current_processed_answer["not_accepted_answer_date"] = top2_unaccepted
        processed_answers.append(current_processed_answer)

    return processed_answers


def store_answers_embeddings(processed_answers: List[Dict]) -> str:
    """
    A function to create and store answer embeddings in Pinecone Vector Store
    :param processed_answers: Processed answers that need to be stored
    :return: A string status indicating if "answers were successfully stored" or "something went wrong with storing answer embeddings"
    """

    answer_strings = []
    for question_answer_pair in processed_answers:
        current_answer_string = f"""\nQuestion Title: {question_answer_pair.get("question_title")}\nQuestion Link: {question_answer_pair.get("question_link")}\n"""

        if question_answer_pair.get("accepted_answer_body"):
            current_answer_string += f"""\nAccepted Answer:\n{question_answer_pair.get("accepted_answer_body")}\n"""

        if question_answer_pair.get("not_accepted_answer_body"):
            for unaccepted_answer in question_answer_pair.get(
                "not_accepted_answer_body"
            ):
                current_answer_string += f"""\nNOT Accepted Answer:\n{unaccepted_answer}\n"""

        answer_strings.append(current_answer_string)

    try:
        pinecone.init(
            api_key=PINECONE_STACK_OVERFLOW_API_KEY,
            environment=PINECONE_STACK_OVERFLOW_ENVIRONMENT
        )

        index_name = PINECONE_STACK_OVERFLOW_INDEX_NAME

        logging.info("Started loading Stack Overflow answers into Pinecone")
        Pinecone.from_texts(answer_strings, embeddings, index_name=index_name)
        logging.info("Successfully loaded Stack Overflow answers into Pinecone")

        return "Answers have been successfully stored."
    except Exception as e:
        logging.error("Something went wrong while storing Stack Overflow answers into Pinecone")
        return "Something went wrong with storing answer embeddings, please try again."


def search_stackoverflow(search_queries: List[str]) -> List[Dict]:
    """
    A function to search for answers in stackoverflow based on search queries
    :param search_queries: A list of search queries that will be used for searching in stackoverflow
    :return: A list of answers. Each element in the list contains question key with question details as well as answer key with answer details
    """

    question_answer_list = []
    question_list = []
    logging.info(f"BEFORE: {search_queries}")
    search_queries = list(set(search_queries))
    logging.info(f"AFTER: {search_queries}")

    for search_query in search_queries:
        questions = SITE.fetch('search/advanced', q=search_query, filter='withbody')

        for question in questions.get("items", []):
            current_question_answer_pair = {}
            if question.get("is_answered"):
                question_id = question.get('question_id')
                accepted_answer_id = question.get('accepted_answer_id')
                if question_id not in question_list:
                    current_question_answer_pair["question"] = {
                        "user_id": question.get("owner").get("user_id"),
                        "profile_link": question.get("owner").get("link"),
                        "answer_view_count": question.get("view_count"),
                        "accepted_answer_id": question.get("accepted_answer_id"),
                        "creation_date": datetime.utcfromtimestamp(question.get("creation_date")).strftime('%Y-%m-%d %H:%M:%S'),
                        "link": question.get("link"),
                        "title": question.get("title"),
                        "body": question.get("body")
                    }

                    if accepted_answer_id:
                        answers = SITE.fetch('questions/{}/answers'.format(question_id), filter='withbody')

                        for answer in answers['items']:
                            answer_details = {
                                "user": answer.get("owner").get("user_id"),
                                "user_reputation": answer.get("owner").get("reputation"),
                                "profile_link": answer.get("owner").get("link"),
                                "is_accepted": answer.get("is_accepted"),
                                "score": answer.get("score"),
                                "creation_date": datetime.utcfromtimestamp(answer.get("creation_date")).strftime('%Y-%m-%d %H:%M:%S'),
                                "body": answer.get("body")}
                            if current_question_answer_pair.get("answers"):
                                current_question_answer_pair["answers"].append(answer_details)
                            else:
                                current_question_answer_pair["answers"] = [answer_details]
                else:
                    logging.info(f"THE FOLLOWING QUESTION WAS ALREADY USED, question id: {question_id}")
                    logging.info(f"THE FULL LIST IS THE FOLLOWING: {question_list}")
            if current_question_answer_pair:
                question_answer_list.append(current_question_answer_pair)
                question_list.append(question_id)

    return question_answer_list


def bm25_ranker(corpus: List[str], query: str, n: int) -> Tuple[List[float], List[str]]:
    """
    A function that uses bm25 algorithm to rerank the provided questions based on their relevance to the given query and choose top n among them
    :param corpus: The corpus or the list of questions that construct the corpus which later will be scored and re-reanked
    :param query: The given query based on which questions must be re-ranked
    :param n: The number of questions that must be returned
    :return: A tuple consisting of scores of each document as well as top-n questions
    """
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()

    doc_scores = bm25.get_scores(tokenized_query)

    top_n = bm25.get_top_n(tokenized_query, corpus, n=n)
    return doc_scores, top_n


def search_stackoverflow_new(search_queries):
    """
    A function to search for answers in stackoverflow based on search queries
    :param search_queries: A list of search queries that will be used for searching in stackoverflow
    :return: A list of answers. Each element in the list contains question key with question details as well as answer key with answer details
    """
    global user_query
    question_id_list = []
    search_queries = list(set(search_queries))
    question_list = []
    question_id_dict = {}
    question_accepted_answer_dict = {}
    question_detail_dict = {}
    full_temp_list = []
    unanswered_question_list = []
    filepath = "ai-agent/stored_question_id_list.json"
    stored_question_id_dict = {}
    stored_question_id_list = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            stored_question_id_dict = json.load(file)
            stored_question_id_list = stored_question_id_dict.get("questions")
    for search_query in search_queries:
        questions = SITE.fetch('search/advanced', q=search_query, filter='withbody')

        for question in questions.get("items", []):

            if question.get("is_answered"):
                question_id = question.get('question_id')
                accepted_answer_id = question.get('accepted_answer_id')
                if question_id not in question_id_list:

                    question_detail_dict[question_id] = {
                        "user_id": question.get("owner").get("user_id"),
                        "profile_link": question.get("owner").get("link"),
                        "answer_view_count": question.get("view_count"),
                        "accepted_answer_id": question.get("accepted_answer_id"),
                        "creation_date": datetime.utcfromtimestamp(question.get("creation_date")).strftime(
                            '%Y-%m-%d %H:%M:%S'),
                        "link": question.get("link"),
                        "title": question.get("title"),
                        "body": question.get("body")
                    }
                    question_id_list.append(question_id)
                    full_question = f"Title: {question['title']}\nBody: {question['body']}"
                    question_list.append(full_question)
                    question_id_dict[full_question] = question_id
                    question_accepted_answer_dict[question_id] = accepted_answer_id

    if question_list:
        scores, top50_questions = bm25_ranker(question_list, user_query, 50)
        final_question_answer_dict = {}
        for top_question in top50_questions:
            current_question_id = question_id_dict.get(top_question)
            if current_question_id:
                final_question_answer_dict[current_question_id] = question_accepted_answer_dict.get(current_question_id)
        for question_id, accepted_answer_id in final_question_answer_dict.items():
            current_question_answer_dict = {}
            if accepted_answer_id:
                if question_id not in stored_question_id_list:
                    answers = SITE.fetch('questions/{}/answers'.format(question_id), filter='withbody')

                    for answer in answers['items']:
                        answer_details = {
                            "user": answer.get("owner").get("user_id"),
                            "user_reputation": answer.get("owner").get("reputation"),
                            "profile_link": answer.get("owner").get("link"),
                            "is_accepted": answer.get("is_accepted"),
                            "score": answer.get("score"),
                            "creation_date": datetime.utcfromtimestamp(answer.get("creation_date")).strftime(
                                '%Y-%m-%d %H:%M:%S'),
                            "body": answer.get("body")}
                        if current_question_answer_dict.get("answers"):
                            current_question_answer_dict["answers"].append(answer_details)
                        else:
                            current_question_answer_dict["answers"] = [answer_details]
                    full_temp_list.append(
                        {"question": question_detail_dict[question_id], "answers": current_question_answer_dict.get("answers")})
            else:
                current_question_details = question_detail_dict.get(question_id)
                if current_question_details:
                    current_question = f"""\nQuestion Title: {current_question_details.get("title")}\nQuestion Link: {current_question_details.get("link")}\n"""
                    unanswered_question_list.append(current_question)

        return full_temp_list, question_id_list, unanswered_question_list
    else:
        return "", "", ""


async def fetch_question_data(client, search_query: str):
    """
    A function to search in stack-overflow for an answer based on the search query
    :param client: Asyncio client instance to conduct an asynchronous call
    :param search_query: A search query that is used for searching an answer in stack-overflow
    :return: Relevant question-answers with their details based on the search query
    """

    response = await client.get('https://api.stackexchange.com/2.3/search/advanced', params={'q': search_query, 'site': 'stackoverflow', 'filter': 'withbody'})
    response.raise_for_status()
    questions = response.json()
    question_answer_list = []

    for question in questions.get("items", []):
        current_question_answer_pair = {}
        if question.get("is_answered"):
            current_question_answer_pair["question"] = {
                "user_id": question.get("owner").get("user_id"),
                "profile_link": question.get("owner").get("link"),
                "answer_view_count": question.get("view_count"),
                "accepted_answer_id": question.get("accepted_answer_id"),
                "creation_date": datetime.utcfromtimestamp(question.get("creation_date")).strftime('%Y-%m-%d %H:%M:%S'),
                "link": question.get("link"),
                "title": question.get("title"),
                "body": question.get("body")
            }
            question_id = question['question_id']
            accepted_answer_id = question.get('accepted_answer_id')
            if accepted_answer_id:
                answers_response = await client.get(f'https://api.stackexchange.com/2.3/questions/{question_id}/answers', params={'site': 'stackoverflow', 'filter': 'withbody'})
                answers_response.raise_for_status()
                answers = answers_response.json()
                for answer in answers['items']:
                    answer_details = {
                        "user": answer.get("owner").get("user_id"),
                        "user_reputation": answer.get("owner").get("reputation"),
                        "profile_link": answer.get("owner").get("link"),
                        "is_accepted": answer.get("is_accepted"),
                        "score": answer.get("score"),
                        "creation_date": datetime.utcfromtimestamp(answer.get("creation_date")).strftime('%Y-%m-%d %H:%M:%S'),
                        "body": answer.get("body")
                    }
                    if current_question_answer_pair.get("answers"):
                        current_question_answer_pair["answers"].append(answer_details)
                    else:
                        current_question_answer_pair["answers"] = [answer_details]
        if current_question_answer_pair:
            question_answer_list.append(current_question_answer_pair)
    return question_answer_list


def search_and_store_answers(search_queries: str) -> str:
    """
    A function that combines previously defined functions to search for answers based on the given search queries in stackoverflow, filter them, and store in Pinecone vector store
    :param search_queries: Search terms/keywords used for searching an answer
    :return: A string status indicating if "answers were successfully stored" or "something went wrong with storing answer embeddings, please try again"
    """

    global global_unanswered_question_list
    search_queries = ast.literal_eval(search_queries)

    initial_answers, question_id_list, unanswered_question_list = search_stackoverflow_new(search_queries)
    if not initial_answers and not question_id_list and not unanswered_question_list:
        return "No questions were found using these keywords, please try again."
    else:
        global_unanswered_question_list = unanswered_question_list

        logging.info(f"SEARCH STACKOVERFLOW RESULTS: {initial_answers}")
        filtered_answers = filter_and_process_answers(initial_answers)

        store_status = store_answers_embeddings(filtered_answers)

        filepath = "ai-agent/stored_question_id_list.json"
        stored_question_id_list = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                stored_question_id_list = json.load(file)
        else:
            logging.error(f"{filepath} FILE DOES NOT EXIST")
        if stored_question_id_list.get("questions"):
            stored_question_id_list["questions"].extend(question_id_list)
        else:
            stored_question_id_list["questions"] = question_id_list
        with open(filepath, 'w') as file:
            json.dump(stored_question_id_list, file)

        return store_status


async def async_generate_keywords(question_list: List[str]) -> List[str]:
    """
    A function that asynchronously generates keywords based on the provided list of question
    :param question_list: A list of questions
    :return: A coroutine object, which, when run, returns a list of keywords for each of the provided questions
    """

    tasks = [keyword_extractor.arun(question) for question in question_list]
    results = await asyncio.gather(*tasks)
    return results


def generate_keywords(question_list: str) -> List[str]:
    """
    A wrapper function to execute keyword generation (async_generate_keywords) based on the provided list of questions
    :param question_list: A list of questions
    :return: A list of keywords for each of the provided questions
    """
    global user_query
    user_query = question_list

    global global_generated_answer
    global_generated_answer = ""
    question_complexity = question_complexity_checker.run(question_list)

    if question_complexity.upper() == "TRUE":
        question_list = question_divider.run(question_list)

    loop = asyncio.get_event_loop()

    question_list = ast.literal_eval(question_list)

    extracted_keyword_list = loop.run_until_complete(async_generate_keywords(question_list))

    extracted_keyword_list = sum([json.loads(extracted_keyword) for extracted_keyword in extracted_keyword_list], [])
    extracted_keyword_list = list(set(extracted_keyword_list))
    return extracted_keyword_list


async def async_score_evidence(evidence_list: List[str], question: str) -> List[str]:
    """
    A function to asynchronously score all of the retrieved evidences (answers from stack-overflow)
    :param evidence_list: A list of Stack-overflow answers retrieved from vector store
    :param question: User given question
    :return: A coroutine object, which, when run, returns a list of numbers from 1 to 5 indicating the usefulness of each answer to the given question, with higher indicating better, or "not useful" if the answer is not useful at all
    """
    logging.info("---"*100)
    for evidence in evidence_list:
        logging.info(f"CURRENT EVIDENCE: {evidence.page_content}")
        logging.info(f"QUESTION: {question}")
    score_tasks = [evidence_scorer.arun(evidence=evidence.page_content, question=question) for evidence in evidence_list]
    results = await asyncio.gather(*score_tasks)
    return results


def score_evidence(evidence_list: List[str], question: str) -> List[str]:
    """
    A wrapper function to execute evidence scoring (async_score_evidence) using the given evidence list and user question
    :param evidence_list: A list of Stack-overflow answers retrieved from vector store
    :param question: User given question
    :return: A list of numbers from 1 to 5 indicating the usefulness of each answer to the given question, with higher indicating better, or "not useful" if the answer is not useful at all
    """
    logging.info("***"*100)

    logging.info(f"QUESTION BEFORE SCORE: {question}")
    logging.info(f"EVIDENCE BEFORE SCORE: {evidence_list}")
    loop = asyncio.get_event_loop()
    evidence_score_list = loop.run_until_complete(async_score_evidence(evidence_list, question))

    return evidence_score_list


def choose_top3_evidences(best_scores: List[int], evidence_score: int, index: int, top_scorer_indices: List[int], final_answer_list: List[str], best_answers: List[str]) -> Tuple[List[str], List[int], List[int]]:
    if evidence_score > best_scores[0] and evidence_score > best_scores[1]:
        if best_scores[0] > best_scores[1]:
            top_scorer_indices[1] = index
            best_scores[1] = evidence_score
            final_answer_list[1] = best_answers[index]
        elif best_scores[1] > best_scores[0]:
            top_scorer_indices[0] = index
            best_scores[0] = evidence_score
            final_answer_list[0] = best_answers[index]
        else:
            index_to_remove = random.choices([0, 1], k=1)[0]
            top_scorer_indices[index_to_remove] = index
            best_scores[index_to_remove] = evidence_score
            final_answer_list[index_to_remove] = best_answers[index]

    elif evidence_score > best_scores[0] and evidence_score < best_scores[1]:
        top_scorer_indices[0] = index
        best_scores[0] = evidence_score
        final_answer_list[0] = best_answers[index]

    elif evidence_score < best_scores[0] and evidence_score > best_scores[1]:
        top_scorer_indices[1] = index
        best_scores[1] = evidence_score
        final_answer_list[1] = best_answers[index]

    elif evidence_score == best_scores[0] and evidence_score == best_scores[1]:
        index_to_remove = random.choices([0, 2], k=1)
        if index_to_remove == 0 or index_to_remove == 1:
            top_scorer_indices[index_to_remove] = index
            best_scores[index_to_remove] = evidence_score
            final_answer_list[index_to_remove] = best_answers[index]

    return final_answer_list, best_scores, top_scorer_indices


def search_answer(question: str) -> List[str]:
    """
    A function to search for relevant answers for the given question in Pinecone using cosine similarity, and then choosing best matches for diversity using Maximum Marginal Relevance (MMR)
    :param question: User given question
    :return: Best answers from Vector Store
    """
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_STACK_OVERFLOW_INDEX_NAME, embedding=embeddings)
    best_answers = docsearch.max_marginal_relevance_search(question, k=15, fetch_k=30)
    return best_answers


def gather_and_check_evidence(best_answers: List[str], question: str) -> str:
    """
    A function that uses search results from vector store to filter them out and leave the ones that are applicable for answering the question.
    :param best_answers: Best answers retrieved from vector store using MMR
    :param question: User question
    :return: Full evidence to answer the question or a message indicating that evidence is not enough
    """
    global global_evidence

    final_answer_list = []
    top_scorer_indices = []
    best_scores = []
    evidence_scores = score_evidence(best_answers, question)
    for index, evidence_score in enumerate(evidence_scores):
        if evidence_score.lower() in ["1", "2", "3", "4", "5"]:
            if len(final_answer_list) < 3:
                final_answer_list.append(best_answers[index])
                best_scores.append(int(evidence_score))
                top_scorer_indices.append(index)
            else:
                final_answer_list, best_scores, top_scorer_indices = choose_top3_evidences(
                    best_scores,
                    int(evidence_score), index,
                    top_scorer_indices,
                    final_answer_list,
                    best_answers
                )

    full_evidence = ""

    for single_evidence in final_answer_list:
        full_evidence += single_evidence.page_content + "\n"
    logging.info(f"FULL EVIDENCE: {full_evidence}")
    evidence_checker_result = evidence_checker.run(evidence=full_evidence, question=question)
    logging.info(f"EVIDENCE CHECKER RESULT: {evidence_checker_result}")
    has_enough_evidence = True if evidence_checker_result.upper() == "TRUE" else False
    if has_enough_evidence:
        global_evidence = full_evidence
        return "Evidence has been successfully gathered"
    else:
        return "Not enough evidence, please gather other results."


def get_evidence(question: str) -> str:
    """
    A function that combines previous functions search_answer, and gather_and_check_evidence to search in vector store and return the most useful evidence
    :param question: The user question
    :return: Full evidence to answer the question or a message indicating that evidence is not enough
    """
    global global_question
    question = global_question
    best_search_results = search_answer(question)
    logging.info(f"BEST SEARCH RESULTS: {best_search_results}")
    evidence = gather_and_check_evidence(best_search_results, question)
    return evidence


def generate_answer(question: str) -> str:
    """
    A function that uses answer generator llm to generate an answer based on the gathered evidence and the given question
    :param question: User given question
    :return: Answer to the user question
    """
    try:
        global global_evidence
        global global_unanswered_question_list
        global global_generated_answer
        generated_answer = answer_generator.run(question=question, evidence=global_evidence, unanswered_question_list=global_unanswered_question_list)
        global_generated_answer = generated_answer

        if generated_answer:
            return "Answer has been successfully generated"
        else:
            return "We were not able to generate the final answer, please try again."
    except Exception as e:
        logging.error(f"Something went wrong while generating an answer: {e}")


def generate_full_response(question: str) -> str:
    """
    A function that uses ai-agent to generate a response to the user question
    :param question: User question
    :return: Generated Response
    """
    global global_question
    global_question = question
    keyword_extractor_tool = Tool.from_function(
        func=generate_keywords,
        name="KeywordExtractor",
        description="You HAVE TO ALWAYS use this tool as your first step whenever you are given a user-question or a list of questions. Used to extract search terms based on the user question that will be later used to find an answer in stackoverflow. The argument to this function should be the exact user question with no modifications. If given multiple questions, DO NOT separate them. Always enclose it in [] to make it in a python list format"
    )

    search_stack_overflow_tool = Tool.from_function(
        func=search_and_store_answers,
        name="SearchStackOverflowAndStore",
        description="Searches in the stackoverflow for an answer using provided search terms. Provide all of the search terms that you have and it will search asynchronously for all of them. Afterwards, it filters the best answers among them and stores their embeddings in Pinecone vector store. This is used to search for an answer later. It returns a string indicating if the data was successfully stored or not"
    )

    gather_evidence_tool = Tool.from_function(
        func=get_evidence,
        name="GatherEvidence",
        description="Searches in our vector store for appropriate answers, filters, and leaves the ones that are useful to answer the question. You either get a message indicating that evidence has been gathered successfully, or a message indicating that there is not enough evidence and you have to start again. The input has to be the original question asked by the user enclosed in square brackets, i.e. [question] "
    )

    generate_answer_tool = Tool.from_function(
        func=generate_answer,
        name="AnswerGenerator",
        description="Takes an input the original question given by the user to generate a well-structured answer that you will use as your result. Your input parameter must be the question specified by the user. The output of this tool is the status indicating if the final answer has been generated successfully or not. If it has been genertaed, finish the chain. No need to output something, just exit the chain with message 'DONE'"
    )

    tools = [keyword_extractor_tool, search_stack_overflow_tool, gather_evidence_tool, generate_answer_tool]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    agent.run(question)
    global global_generated_answer
    return global_generated_answer
