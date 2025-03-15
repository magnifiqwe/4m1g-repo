import openai
import logging
from config import *
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from handlers import register_all_handlers

openai.api_key = OPENAI_API
openai.base_url = 'https://api.vsegpt.ru/v1/'


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(__name__).error("Starting bot")
    bot = Bot(BOT_API)
    dp = Dispatcher(bot, storage=MemoryStorage())

    register_all_handlers(dp)

    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
