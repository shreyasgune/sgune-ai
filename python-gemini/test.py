import google.generativeai as genai
import os, sys, getopt

envkey = os.getenv("GEMINI_API_KEY")
# GOOGLE_API_KEY = userdata.get(envkey)
genai.configure(api_key=envkey)

model = genai.GenerativeModel('gemini-1.5-flash')

question = sys.argv[1]

response = model.generate_content(question)

print(response)