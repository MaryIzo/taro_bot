# Проект

Найти бот можно в телеграме по https://t.me/slemary_taro_bot    
/start - запустить бота   
Видео-инструкция есть здесь: IMG_2600.MP4    
Если бот не отвечает, напишите мне в телеграм @hellowtriangle, но он должен работать непрерывно  

https://github.com/MaryIzo/taro_bot/tree/master   

https://colab.research.google.com/drive/1fbu1NBi_hrWDR0kJu32jvdVpFd8tugXA?usp=sharing   

## Название проекта

Taro bot

## Запуск   

    /start - запустить бота
    /help - вызвать меню команд
    
    /question_about_taro - задать вопрос о картах таро (использование RAG)
    
    /yes_or_no  - ответить на вопрос да или нет
    /one_card_question - ответить на вопрос на основе выпавшей карты
    /one_card - выложть карту на день 
    /past_present_future - прошлое настоящее и будущее в вашем вопросе   
    /the_celtic_cross - расклад Кельтский крест   
    
    /simple_question - задать любой вопрос   


## Основная идея, уникальность, целевая аудитория

Часто бывает такое, что нужно принять какое-то важдое решение и "посоветоваться" с судьбой. Основываясь вероятностях и случайности и возможности человека обратиться с каким-то вопросом к искуственному интеллекту.  
Последнее время в интернете стали очень популярны гадания на картах таро, которые окружают себя ореолом таинственности, но по сути основаны на случайном выпадении карт в ответ на ваши вопросы.   
Давайте попробуем повторить это.  
Основная аудитория достаточно широкая.   
Приложение может быть популярно, как печенье предсказаний для тех, кто любит гадать.    
А ак же может отвечать на вопросы для тех кто интересуется как устроено само гадание.   
    
## Особенности реализации (основные фишки, дальнейшие шаги для улучшения)

1.  Добавим в модель промт, который превращает ее в гадалку
2.  Разобьем все возможные гадания на несколько вариантов: карта на день, прошлое настоящее будущее, вопрос, кельтский крест. От типа расклада зависит количество карт.    
3.  Используем RAG и случайность для того чтобы возвращать ответ на вопрос, метрика mmr.  
4.  Надо разбить текст на части в которых есть:
     - описание каждого типа расклада
     - описание всех возможных карт
     - картинки всех возможных карт (можно попробовать использовать мультимодальную модель потом)

В качестве улучшения переделать на русский язык, добавить мультимодальную модель и сделать кнопочки.

## Необходимые библиотеки и ресурсы для работы сервиса

Библиотеки в requirements.txt.   
Необходимо подключаться на хост для работы с ботом.      

## Технические особенности (сбор и работа с данными, техники промптинга, модели, хранилища и.т.п.)

Данные разбиваются вручную и по абзацам, это дало наиботльшее качество результата.  
В промптинге ллм должна представить себя гадалкой.   
В качестве хранилища FAISS, модель Mistral, gemma работала лучше, но она не отвечает по API,
а у меня нет достаточного количества ресурсов на локальную ллм.  

## Деплой (где развернули приложение, почему, особенности взаимодействия с пользователем)
Тест: https://t.me/slemary_taro_bot      

/start - запустить бота   
/help - вызвать меню команд   
    
python3 taro_bot.py -u 1>>stdout.txt 2>>stderr.txt &     
ps aux | grep python    
kill -9 8758   