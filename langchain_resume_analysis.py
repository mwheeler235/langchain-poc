import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

os.environ.get("OPENAI_API_KEY")

reader = PdfReader('resumes/Matt Wheeler Resume 2024.pdf')
with open('resumes/txt/mw_resume_2024.txt', 'w') as f:
    f.write(reader.pages[0].extract_text())

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

with open('resumes/txt/mw_resume_2024.txt') as f:
    resume = f.read()

# Create prompt
q = f'''
You are a hiring manager reviewing a candidate resume for a marketing senior data scientist position.


Extract specified values from the source text.
Return answer as JSON object with following fields:
- "candidate_name" <string>
- "candidate_deep_passions" <string>
- "what_are_the_biggest_topics_in_this_resume" <string>
- "candidate_programming_languages" <string>
- "python_libraries_used" <string>
- "does_candidate_have_ml_experience" <string>
- "does_candidate_have_llm_experience" <string>
- "does_candidate_have_web_design_experience" <string>
- "does_candidate_have_data_visualization_experience" <string>
- "candidate_industries_with_experience" <string>
- "is_candidate_good_fit_for_position" <string>
- "is_candidate_a_cat" <string>


Do not infer any data based on previous training, strictly use only source text given below as input.
========
{resume}
========
'''

# OpenAI call, print result
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": q,
        }
    ],
    model="gpt-3.5-turbo",
)
llm_response = chat_completion.choices[0].message.content
print(llm_response)