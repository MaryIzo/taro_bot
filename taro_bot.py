import asyncio
import os

from aiogram.utils import executor
from aiogram import Bot, Dispatcher, executor, types
from langchain_community.vectorstores.faiss import FAISS
import pickle
import requests
import random


STATE = None

# TG_TOKEN = TG_TOKEN_PATH.read_text().strip()
TG_TOKEN = '7431248846:AAGT3wbL8IbdumwvdLH1ZwWHbmftV6sL2Pw'
os.environ['VERBOSE'] = 'True'


# ------  Initialize bot and dispatcher --------------

bot = Bot(token=TG_TOKEN)
dispatcher = Dispatcher(bot)

# --------  LOAD DATABASE AND RETRIEVER --------------------

with open('cards_meaning_splits.pkl', 'rb') as f:
    cards_meaning_splits = pickle.load(f)

with open('hf_embeddings_model.pkl', 'rb') as f:
    loaded_embeddings_model = pickle.load(f)

db = FAISS.load_local(
    "faiss_db", loaded_embeddings_model, allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr",  # тип поиска похожих документов
    k=1,  # количество возвращаемых документов (Default: 4)
    score_threshold=0.9,  # минимальный порог для поиска "similarity_score_threshold"
)

# --------------- FUNCTIONS ----------------

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

def get_one_card_prediction(random_card, topic=''):
  """
  Функция забирает суммаризацию на предсказание одной карты.
  Здесь лучше всего работает mistral.
  """
  api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

  additional_prompt = """
      Find a peaceful, relaxed feeling. Feel comfortable and confident.
      Imagine that you can hear the universe and are a good fortune teller.
      Make only three sentences.
      Try to use information:
      {}
      """.format(' '.join(random_card))
  main_input = "What is your Summirization? {}".format(topic)

  return get_query(additional_prompt, main_input)

def get_yes_or_no(test_cards):
  """
  Функция  отвечает на вопрос Да или Нет.
  """
  api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

  additional_prompt = """
      If you get three yeses, your answer is certain; two yes cards
      (with one no or neutral) mean a positive outcome is most likely, but it may take
      time to come about; and all no or a mix of no/neutral cards means the answer to
      your question is, of course, negative.

      Yes cards: All cards apart from those listed here as no, neutral, or exceptions.

      No cards
      Swords: Three, Five, Six, Seven, Eight, Nine, Ten, and Knight
      Cups: Five, Seven, and Eight
      Pentacles: Five
      Death
      The Devil
      The Tower
      The Moon

      Neutral cards
      Swords: Four
      Cups: Four
      The Hermit
      The Hanged Man

      Exceptions
      Two of Swords or Ten of Wands: The answer is not yet known.
      Five and Seven of Wands: The answer is yes, but you must fight for your
      prize.
      Your cards are:
      {}
      """.format(' '.join(test_cards))
  main_input = "What is your answer, yes or no? And why?"

  return get_query(additional_prompt, main_input)

# ------------------ MESSAGE HAMDLER ----------------------

@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    A handler to welcome the user and clear past conversation and context.
    """
    await message.reply("Hello! \n I can help you with taro and prediction!\
                        \nWhat would you like to do?")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Hello! I am taro chat-bot. Please use the following commands:
    /start - start a bot
    /help - ask for help
    
    /question_about_taro - ask about taro 
    
    /yes_or_no  - yes or no answer on your question
    /one_card_question - answer on one card
    /one_card - one day card 
    /past_present_future - past, presend and future for your question
    /the_celtic_cross - celtic cross for you
    
    /simple_question - ask any question
    """
    await message.reply(help_command)


@dispatcher.message_handler(commands=[
    'simple_question', 
    'one_card', 
    'question_about_taro',
    "one_card_question", 
    "past_present_future", 
    "the_celtic_cross", 
    "yes_or_no"
    ])
async def define_state(message: types.Message):
    """
    A handler to process the user's input and generate a response using the chatGPT API.
    """
    global STATE
    print(f">» USER: \n{message.text}")
    if message.text == '/simple_question':
        STATE = 'simple_question'
        await bot.send_message(chat_id=message.chat.id, text='Write your question about anyting')
    elif message.text == '/one_card':
        STATE = 'one_card'
        await bot.send_message(chat_id=message.chat.id, text='Write any symbol')
    elif message.text == '/question_about_taro':
        STATE = 'question_about_taro'
        await bot.send_message(chat_id=message.chat.id, text='Write your question about taro')
    elif message.text == "/one_card_question":
        STATE = 'one_card_question'
        await bot.send_message(chat_id=message.chat.id, text='Write your question about anyting')
    elif message.text == "/past_present_future":
        STATE = 'past_present_future'
        await bot.send_message(chat_id=message.chat.id, text='Write your question')
    elif message.text ==  "/the_celtic_cross":
        STATE = 'the_celtic_cross'
        await bot.send_message(chat_id=message.chat.id, text='Write your question')
    elif message.text ==  "/yes_or_no":
        STATE = 'yes_or_no'
        await bot.send_message(chat_id=message.chat.id, text='Write your question for yes or no')

@dispatcher.message_handler()
async def get_question(message: types.Message):
    
    global STATE

    if STATE == "question_about_taro":
        
        retriver_answer = retriever.get_relevant_documents(
            message.text
        )
        retriver_str_answer = [split.page_content for split in retriver_answer]
        answer = get_retriever(retriver_str_answer[0], message.text)

    elif STATE == 'one_card':

        random_key = random.choice(list(cards_meaning_splits.keys()))
        additional_prompt = "One card for your events: "
        random_card = '\n'.join(cards_meaning_splits[random_key])
        answer = ' '.join([additional_prompt, random_card])

    elif STATE == "one_card_question":

        random_key = random.choice(list(cards_meaning_splits.keys()))
        random_card = ' '.join(cards_meaning_splits[random_key])
        answer = get_one_card_prediction(
            random_card,
            f' Answer on question {message.text}.'
        )            

    elif STATE == "past_present_future":

        times = {0: 'past', 1: 'present', 2: 'future'}
        answer = []
        for key, val in times.items():
            random_key = random.choice(list(cards_meaning_splits.keys()))
            random_card = ' '.join(cards_meaning_splits[random_key])
            card_answer = get_one_card_prediction(random_card, f' Answer in {val} tense.')
            answer.append( f"""{val}: {card_answer}""")

        answer = '\n'.join(answer)

    elif STATE == "the_celtic_cross":

        questions = {
            0: "1 Your situation",
            1: "2 Responsibilities",
            2: "3 Limitations and the past",
            3: "4 What supports you",
            4: "5 What opposes you",
            5: "6 Achievements",
            6: "7 Attraction and relationships",
            7: "8 Work, health, and communication",
            8: "9 What is hidden",
            9: "10 The future environment; the outcome"
        }
        for key, val in questions.items():
            random_key = random.choice(list(cards_meaning_splits.keys()))
            random_card = ' '.join(cards_meaning_splits[random_key])

            answer = []
            card_answer = get_one_card_prediction(random_card, f' Answer on question {val}.')
            answer.append( f"""{val}: {card_answer}""")

        answer = '\n'.join(answer)

    elif STATE == "yes_or_no": 

        test_cards = []
        for i in range(3):
            test_cards.append(random.choice(list(cards_meaning_splits.keys())))

        answer = get_yes_or_no(test_cards)

    elif STATE == 'simple_question':

        additional_prompt = """
            Find a peaceful, relaxed feeling. Feel comfortable and confident.
            Imagine that you can hear the universe and are a good fortune teller.
            Answer only thre sentences.
        """
        answer = get_query(additional_prompt, message.text)

    else:
        answer = 'Something went wrong. Try again.'

    print(f">» llm: \nwriting...")
    await bot.send_message(chat_id=message.chat.id, text='writing...')

    print(f">» llm: \n{answer}")
    await bot.send_message(chat_id=message.chat.id, text=f"{answer}")

    await bot.send_message(chat_id=message.chat.id, text="Write new command.")
    
    print(message.text)

executor.start_polling(dispatcher, skip_updates=True)


# if __name__ == '__main__':
#     print("Starting...")
#     executor.start_polling(dispatcher, skip_updates=True)
#     print("Stopped")