from openai import OpenAI
from .prompt import UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT, NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT, UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE,NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE, EXTRACT_OPTION_HUMAN_PROMPT_TEMPLATE,   \
UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE, NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
import requests
import json

import aiohttp
from openai import AsyncOpenAI  # Use official async client

from tenacity import retry, stop_after_attempt, wait_exponential


# Using proxy URLs with specific APIs eliminates the need to send requests to OpenAI directly
# Usage is consistent with OpenAI official API, only need to modify baseurl
# Baseurl = "https://api.claudeshop.top"
# Skey = "sk-so7zhdci3gxgggAyXM3UxevY0XsJGlzUzOXyMgdcQswplf7T"

class BaseModel:
    def __init__(self, model='gpt-4o', api_key='your_api_key', base_url='your_base_url', separate=False):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        # self.url = base_url + "/v1"#chat/completions"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # self.multi_round_system_prompt = MULTI_ROUND_SYSTEM_PROMPT
        
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
            }
        
    
        self.url = base_url + "/chat/completions"

        self.eval_unique_answer_system_prompt = UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
        self.eval_unique_answer_system_prompt_separate = UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
        self.eval_non_unique_answer_system_prompt = NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
        self.eval_non_unique_answer_system_prompt_separate = NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
        self.eval_unique_answer_human_prompt_template = UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
        self.eval_non_unique_answer_human_prompt_template = NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
        self.extract_option_huamn_prompt_template = EXTRACT_OPTION_HUMAN_PROMPT_TEMPLATE
        
        self.separate = separate

    def evaluate_unique_answer_response(self, question, response, answer, procedure):
        # Call OpenAI ChatCompletion API
        if self.separate:
            message_record = [
                {"role": "system", "content": self.eval_unique_answer_system_prompt_separate},
                {"role": "user", "content": self.eval_unique_answer_human_prompt_template.format(question=question, response=response, answer=answer, procedure=procedure)}
            ]
        else:
            message_record = [
                {"role": "system", "content": self.eval_unique_answer_system_prompt},
                {"role": "user", "content": self.eval_unique_answer_human_prompt_template.format(question=question, response=response, answer=answer, procedure=procedure)}
            ]
 
        payload = json.dumps({
            "model": self.model,
            "messages": message_record
        })
        
        eval_response = requests.request("POST", self.url, headers=self.headers, data=payload, timeout=100)
        

        # print(eval_response.text)
        try:
            eval_response = eval_response.json()['choices'][0]['message']['content']
        except (requests.Timeout, requests.ConnectionError) as e:
            raise

        return eval_response
    
    def evaluate_non_unique_answer_response(self, summary, question, response, answer, procedure):
        limit = 10000
        if len(summary) > limit: summary = summary[:limit]

        if self.separate:
            message_record = [
                {"role": "system", "content": self.eval_non_unique_answer_system_prompt_separate},
                {"role": "user", "content": self.eval_non_unique_answer_human_prompt_template.format(video_summary = summary, question=question, response=response, procedure=procedure, answer=answer)}
            ]
        else:
            message_record = [
                {"role": "system", "content": self.eval_non_unique_answer_system_prompt},
                {"role": "user", "content": self.eval_non_unique_answer_human_prompt_template.format(video_summary = summary, question=question, response=response, procedure=procedure, answer=answer)}
            ]
        
        payload = json.dumps({
            "model": self.model,
            "messages": message_record
        })

        try:

            eval_response = requests.request("POST", self.url, headers=self.headers, data=payload, timeout=100, verify=False)
        # print(eval_response.text)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise
    
        eval_response = eval_response.json()['choices'][0]['message']['content']

        return eval_response
    
    def extract_option_from_answer(self, multiple_choice_question_answer):
        message_record = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.extract_option_huamn_prompt_template.format(multiple_choice_question_answer = multiple_choice_question_answer)}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=message_record
        )
        # Return response content
        response = completion.choices[0].message.content
              
        return response 
    


class AsyncBaseModel:
    def __init__(self, model='gpt-4o', api_key='your_api_key', base_url='your_base_url', separate=False):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.separate = separate
      
        # Initialize async client
        import httpx
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(verify=False)
        )
      
        # Other configurations remain unchanged
        self.eval_unique_answer_system_prompt = UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
        self.eval_unique_answer_system_prompt_separate = UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
        self.eval_non_unique_answer_system_prompt = NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
        self.eval_non_unique_answer_system_prompt_separate = NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
        self.eval_unique_answer_human_prompt_template = UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
        self.eval_non_unique_answer_human_prompt_template = NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
        self.extract_option_huamn_prompt_template = EXTRACT_OPTION_HUMAN_PROMPT_TEMPLATE

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def evaluate_unique_answer_response(self, question, response, answer, procedure):
        """Async evaluation for unique answer type responses"""
        try:
            # Construct message record
            if self.separate:
                system_prompt = self.eval_unique_answer_system_prompt_separate
            else:
                system_prompt = self.eval_unique_answer_system_prompt
              
            message_record = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.eval_unique_answer_human_prompt_template.format(
                    question=question, 
                    response=response, 
                    answer=answer, 
                    procedure=procedure
                )}
            ]
          
            # Async API call
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=message_record,
                timeout=30.0  # Set timeout
            )
            return completion.choices[0].message.content
          
        except Exception as e:
            print(f"Evaluate unique answer error: {str(e)}")
            return ""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def evaluate_non_unique_answer_response(self, summary, question, response, answer, procedure):
        """Async evaluation for non-unique answer type responses"""
        try:
            # Handle long summaries
            limit = 10000
            if len(summary) > limit: 
                summary = summary[:limit]
              
            # Select system prompt
            if self.separate:
                system_prompt = self.eval_non_unique_answer_system_prompt_separate
            else:
                system_prompt = self.eval_non_unique_answer_system_prompt
              
            # Construct messages
            message_record = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.eval_non_unique_answer_human_prompt_template.format(
                    video_summary=summary,
                    question=question,
                    response=response,
                    procedure=procedure,
                    answer=answer
                )}
            ]
          
            # Async call
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=message_record,
                timeout=30.0
            )
            return completion.choices[0].message.content
          
        except Exception as e:
            print(f"Evaluate non-unique answer error: {str(e)}")
            return ""

    async def extract_option_from_answer(self, multiple_choice_question_answer):
        """Async extract option from response"""
        try:
            message_record = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.extract_option_huamn_prompt_template.format(
                    multiple_choice_question_answer=multiple_choice_question_answer
                )}
            ]
          
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=message_record,
                timeout=20.0
            )
            return completion.choices[0].message.content
          
        except Exception as e:
            print(f"Extract option error: {str(e)}")
            return ""
    

# Example usage
if __name__ == "__main__":
    print("hello,world")