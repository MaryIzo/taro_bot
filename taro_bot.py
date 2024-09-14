import os

from aiogram import Bot, Dispatcher, executor, types
from langchain.vectorstores import FAISS
import pickle

TG_TOKEN = TG_TOKEN_PATH.read_text().strip()

os.environ['VERBOSE'] = 'True'


# ------  Initialize bot and dispatcher --------------
bot = Bot(token=TG_TOKEN)
dispatcher = Dispatcher(bot)

# --------  LOAD DATABASE AND RETRIEVER --------------------

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

# ----------------MY CODE STARTS HERE----------------

def clear_past():
    """
    A function to clear the previous conversation and context.
    """
    reference.response = ''


@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    A handler to welcome the user and clear past conversation and context.
    """
    clear_past()
    await message.reply("Hello! \n I can help you with taro and predictiona!\
                        \nWhat would you like to do?")


@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the previous conversation and context.
    """
    clear_past()
    await message.reply("I cleared last conversation and context.")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Hello! I am taro chat-bot. Please use the following commands:
    /start - 
    /clear - clear conversation and context
    /help - ask for help
    """
    await message.reply(help_command)


@dispatcher.message_handler()
async def question_about_taro(message: types.Message):
    """
    A handler to process the user's input and generate a response using the chatGPT API.
    """
    print(f">» USER: \n{message.text}")

    # ----------------MY CODE STARTS HERE----------------


    retriver_answer = retriever.get_relevant_documents(
        message.text
    )
    retriver_str_answer = [split.page_content for split in retriver_answer]
    answer = get_retriever(retriver_str_answer[0], message.text)

    # ----------------MY CODE ENDS HERE----------------

    print(f">» llm: \nwriting...")
    await bot.send_message(chat_id=message.chat.id, text='writing...')

    print(f">» llm: \n{answer}")
    await bot.send_message(chat_id=message.chat.id, text=f"{answer}")

if __name__ == '__main__':
    print("Starting...")
    executor.start_polling(dispatcher, skip_updates=True)
    print("Stopped")