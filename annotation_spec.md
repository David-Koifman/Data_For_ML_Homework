# Спецификация разметки — sentiment_classification

*Сгенерировано: 2026-03-26 15:00*

# Спецификация разметки: Классификация тональности текстов (Sentiment Classification)

## 1. Описание задачи

Задача заключается в определении эмоциональной окраски (тональности) текстовых рецензий на фильмы. Разметчик должен отнести каждый текст к одному из трёх классов: позитивная, негативная или нейтральная тональность.

**Целевое применение:** классификация отзывов пользователей для рекомендательных систем, анализа репутации контента, агрегации оценок.

**Язык текстов:** преимущественно английский с возможными опечатками и HTML-сущностями.

---

## 2. Определения классов

### POSITIVE (Позитивная тональность)

**Определение:** Текст выражает похвалу, одобрение и положительное отношение к фильму. Автор рецензии рекомендует фильм или считает его достойным внимания.

**Ключевые маркеры:**
- Слова одобрения: "favorite", "stunning", "brilliant", "amazing", "excellent", "worthy"
- Рекомендации: "worth watching", "must see", "highly recommend"
- Позитивные описания фильма как целого
- Высокая оценка актёрской игры, режиссуры, сценария
- Эмоции восхищения и удовлетворения

**Примеры:**

1. *"I dug out from my garage some old musicals and this is another one of my favorites. It was written by Jay Alan Lerner and directed by Vincent Minelli. It won two Academy Awards for Best Picture of 195"*
   - Маркеры: "one of my favorites", упоминание Academy Awards, ностальгический тон

2. *"'The Luzhin Defence' is a movie worthy of anyone's time. it is a brooding, intense film, and kept my attention the entire time. John Turturro is absolutely stunning in his portrayal of a tender, eccen"*
   - Маркеры: "worthy of anyone's time", "stunning", "kept my attention"

3. *"Just like Al Gore shook us up with his painfully honest and cleverly presented documentary-movie "An inconvenient truth", directors Alastair Fothergill and Mark Linfield also remind us that it's about"*
   - Маркеры: "cleverly presented", сравнение с успешным фильмом, позитивные прилагательные

---

### NEGATIVE (Негативная тональность)

**Определение:** Текст содержит критику, неодобрение и отрицательное отношение к фильму. Автор выражает разочарование, скуку или несогласие с качеством кинопроизведения.

**Ключевые маркеры:**
- Слова критики: "dumb", "uninteresting", "uninspired", "disappointed", "boring", "terrible", "waste"
- Явное выражение разочарования: "disappointed", "let down", "regret"
- Критика аспектов: "bad acting", "poor directing", "weak plot"
- Отрицательные сравнения
- Рекомендация НЕ смотреть

**Примеры:**

1. *"Dumb is as dumb does, in this thoroughly uninteresting, supposed black comedy. Essentially what starts out as Chris Klein trying to maintain a low profile, eventually morphs into an uninspired version"*
   - Маркеры: "Dumb", "thoroughly uninteresting", "uninspired"

2. *"After watching this movie I was honestly disappointed - not because of the actors, story or directing - I was disappointed by this film advertisements.<br /><br />The trailers were suggesting that the"*
   - Маркеры: "honestly disappointed" (повторено), критика маркетинга

3. *"This movie was nominated for best picture but lost out to Casablanca but Paul Lukas beat out Humphrey Bogart for best actor. I don't see why Lucile Watson was nominated for best supporting actor, i ju"*
   - Маркеры: сравнение не в пользу (проиграло), сомнение в обоснованности номинаций

---

### NEUTRAL (Нейтральная тональность)

**Определение:** Текст содержит преимущественно фактическую информацию о фильме (сюжет, актёры, года создания) без явного выражения положительных или отрицательных оценок. Автор описывает, но не судит.

**Ключевые маркеры:**
- Преобладание фактической информации (режиссер, актеры, год выпуска)
- Описание сюжета без оценок
- Сбалансированное изложение достоинств и недостатков
- Отсутствие оценочных прилагательных
- Информационный стиль

**Примеры:**

1. *"This film was directed by Martin Scorsese and stars Leonardo DiCaprio. The movie was released in 2013 and has a runtime of 180 minutes"*
   - Маркеры: только фактические данные

2. *"The main character is played by an experienced actor. The film combines elements of drama and thriller. It was nominated for several awards but did not win"*
   - Маркеры: описание без оценки

3. *"Based on the novel by Jane Austen, the film features Emma Thompson as the lead. Cinematography was handled by John Williams, and the score was composed by Patrick Doyle"*
   - Маркеры: фактическая информация о составе

---

## 3. Граничные случаи (Edge Cases