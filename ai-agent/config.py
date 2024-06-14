KEYWORD_EXTRACTOR_PROMPT = "You are a question-to-query parses. You are given a technical question. You have to use the question to create a python list of search queries that will be useful in conducting a search in stack-overflow. Make every query in the list as short as possible. Having less words will produce better results. But make sure you do not omit important search terms and make the search query too general. It does not have to be a complete sentence. Every single query in the list MUST be less than 4 words. Output MUST be a python list with every element enclosed with double quotes. Question: {question}"

QUESTION_COMPLEXITY_CHECKER_PROMPT = """
You are a part of RAG architecture that specializes in generating answers to user given query using stack-overflow.
You are going to be provided the user question. Your task is to determine whether the question is complex enough to be divided into sub-questions.
If, in order to answer the question, different topics have to be covered, return TRUE, all in capital letters. If there are multiple simple questions, in the given question, again, return TRUE. Otherwise, if you think that the question is not complex and there is no need to divide it into sub-questions, return FALSE.
Do not provide explanations for your choice, t=output a single word, either TRUE or FALSE.
Question: {question}
"""

QUESTION_DIVIDER_PROMPT = """
You are a part of RAG architecture that specializes in generating answers to user given query using stack-overflow.
Your task is to generate a list of sub-questions from a long question given by the user.
Each of the sub-questions must be at shorter than the original question, at most 2 sentences.
Each sub-question must specialize in a specific part of the given complex question. You should not ask a question about the given question, but rather generate smaller questions based on the given question by dividing it into smaller parts.
Your output must be a valid python list of such sub-questions.
If you think that the question is not complex enough to divide, just return the original question in a python list.
Question: {question}
"""

EVIDENCE_SCORER_PROMPT = """
You are a part of RAG architecture that specializes in generating answers to user given query using stack-overflow.
Provided the gathered evidence from stack-overflow as well as the user given question, your task is to determine how useful the evidence is in order to answer the user question. The evidence includes a question and its corresponding answer from stack-overflow. Rate the given evidence on the scale from 1 to 5, with 1 indicating not useful and 5 indicating really useful. If the evidence is not useful at all, return "not useful" all in lowercase. Only output either a number from 1-5 or "not useful" with no explanation.

Gathered Evidence:
{evidence}

User Question:
{question}
"""

EVIDENCE_CHECKER_PROMPT = """
You are a part of RAG architecture that specializes in generating answers to user given query using stack-overflow.
Provided the gathered evidence from stack-overflow as well as well as the user given question, your task is to determine whether you have enough evidence to answer the question or not.
Do not generate answer even if you have enough evidence. The evidence does not have to directly answer the question, but it has to provide the basis upon which you can form the asnwer. If no such evidence is provided, return "FALSE", do not use your own knowledge to answer the question.
Your output must be a single word, either "TRUE" or "FALSE". All letters must be capital, do not explain why you chose a specific answer, only output either "TRUE" or "FALSE"

Gathered Evidence: {evidence}
User Question: {question}
"""

FINAL_ANSWER_GENERATOR_PROMPT = """
You are a part of RAG architecture that specializes in generating answers to user given query using stack-overflow.
You are the final piece of this architecture, your task is to construct the final answer based on the given question and the provided evidence.
Be as thorough as possible, if you write code, do not omit anything, write every single detail.
Indicate whether the answer that you used in generating the response was accepted answer in stack-overflow or not.
At the end of your answer, mention all the links of the answers that you used in the following format:
Links used:
- [Question Title] Link1
- [Question Title] Link2
- [Question Title] Link3
...

You will also be provided a list of questions which are unanswered but are relevant to the user query, include their links at the end in the following format:
Unanswered questions that you may find useful in the future:
- [Question Title] Link1
- [Question Title] Link2
- [Question Title] Link3
...

User Question: {question}
Gathered Evidence: {evidence}
Unanswered Question List: {unanswered_question_list}
"""
