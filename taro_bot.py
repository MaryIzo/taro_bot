import asyncio
import os

from aiogram.utils import executor
from aiogram import Bot, Dispatcher, executor, types
from langchain_community.vectorstores.faiss import FAISS
import pickle
import requests

# TG_TOKEN = TG_TOKEN_PATH.read_text().strip()
TG_TOKEN = '7431248846:AAGT3wbL8IbdumwvdLH1ZwWHbmftV6sL2Pw'
os.environ['VERBOSE'] = 'True'


# ------  Initialize bot and dispatcher --------------

bot = Bot(token=TG_TOKEN)
dispatcher = Dispatcher(bot)

# --------  LOAD DATABASE AND RETRIEVER --------------------

# with open('hf_embeddings_model.pkl', 'rb') as f:
#     loaded_embeddings_model = pickle.load(f)

# db = FAISS.load_local(
#     "faiss_db", loaded_embeddings_model, allow_dangerous_deserialization=True
# )

# retriever = db.as_retriever(
#     search_type="mmr",  # тип поиска похожих документов
#     k=1,  # количество возвращаемых документов (Default: 4)
#     score_threshold=0.9,  # минимальный порог для поиска "similarity_score_threshold"
# )

# ---------------ADDITIONAL FUNCTIONS ----------------

def get_query(additional_prompt, main_input):
  """Function that returns response from hugging face API open model.
  """

  # Combine the additional prompt and main input with a special token
  query = f"{additional_prompt} [END_PROMPT] {main_input}"

  api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

  headers = {
      "Authorization": "Bearer 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",
      "Content-Type": "application/json"
  }

  data = {
      "inputs": query,
      "parameters": {
          "max_length": 10000,  # Adjust as needed
          "temperature": 0.9  # Adjust as needed
      }
  }

  # Send the API request
  response = requests.post(api_url, headers=headers, json=data)

  # Parse the response
  if response.status_code == 200:
      output = response.json()
      # Extract the main output, excluding the additional prompt
      main_output = output[0]['generated_text'].split("[END_PROMPT]")[1].strip()
      main_output = main_output.replace(main_input, '').replace('\n', '').strip()
      main_output = main_output.split('.')[:3]

  else:
      print(f"Error: {response.status_code} - {response.text}")

  return '.'.join(main_output)


def get_retriever(retriever_answer, question='What can you tell about taro?'):
  """
  Функция забирает суммаризацию на предсказание одной карты.
  Здесь лучше всего работает mistral.
  """


  additional_prompt = """
      Imagine that you know everything about taro. Return only the detailed answer.
      Use the following information if it is useful:
      {}
      """.format(' '.join(retriever_answer))
  main_input = "{}".format(question)

  return get_query(additional_prompt, main_input)

# ----------------MY CODE STARTS HERE----------------


@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    A handler to welcome the user and clear past conversation and context.
    """
    await message.reply("Hello! \n I can help you with taro and predictiona!\
                        \nWhat would you like to do?")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Hello! I am taro chat-bot. Please use the following commands:
    /start - start a bot
    /clear - clear conversation and context
    /help - ask for help
    /simple_question - ask any question
    /question_about_taro - ask about taro 
    """
    await message.reply(help_command)

STATE = None

@dispatcher.message_handler(commands=['simple_question', 'question_about_taro'])
async def define_state(message: types.Message):
    """
    A handler to process the user's input and generate a response using the chatGPT API.
    """

    print(f">» USER: \n{message.text}")
    if message.text == '/simple_question':
        global STATE
        STATE = 'simple_question'
        await bot.send_message(chat_id=message.chat.id, text='Write your question')
    


@dispatcher.message_handler()
async def get_question(message: types.Message):
    
    if STATE == 'simple_question':

        # ----------------MY CODE STARTS HERE----------------
        additional_prompt = """
        Find a peaceful, relaxed feeling. Feel comfortable and confident.
        Imagine that you can hear the universe and are a good fortune teller.
        Answer only thre sentences.
        """
        answer = get_query(additional_prompt, message.text)

        # ----------------MY CODE ENDS HERE----------------

    print(f">» llm: \nwriting...")
    await bot.send_message(chat_id=message.chat.id, text='writing...')

    print(f">» llm: \n{answer}")
    await bot.send_message(chat_id=message.chat.id, text=f"{answer}")
    
    print(message.text)

executor.start_polling(dispatcher, skip_updates=True)

# @dispatcher.message_handler(commands=['question_about_taro'])
# async def question_about_taro(message: types.Message):
#     """
#     A handler to process the user's input and generate a response using the chatGPT API.
#     """
#     print(f">» USER: \n{message.text}")

#     # ----------------MY CODE STARTS HERE----------------
#     retriver_answer = retriever.get_relevant_documents(
#         message.text
#     )
#     retriver_str_answer = [split.page_content for split in retriver_answer]
#     answer = get_retriever(retriver_str_answer[0], message.text)

#     # ----------------MY CODE ENDS HERE----------------

#     print(f">» llm: \nwriting...")
#     await bot.send_message(chat_id=message.chat.id, text='writing...')

#     print(f">» llm: \n{answer}")
#     await bot.send_message(chat_id=message.chat.id, text=f"{answer}")



# if __name__ == '__main__':
#     print("Starting...")
#     executor.start_polling(dispatcher, skip_updates=True)
#     print("Stopped")