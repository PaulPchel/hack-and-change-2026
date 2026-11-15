import unittest
from unittest.mock import patch

import consultant


class ConsultantTest(unittest.TestCase):
    def turn(self, history, text):
        history.append({"role": "user", "content": text})
        reply = consultant.answer(history)
        history.append({"role": "assistant", "content": reply["text"]})
        return reply

    def test_lead_from_history_and_no_duplicate(self):
        history = []
        self.assertEqual(self.turn(history, "Перезвоните мне")["text"], consultant.QUESTIONS["name"])
        self.assertEqual(self.turn(history, "Иванов Иван Иванович")["text"], consultant.QUESTIONS["phone"])
        self.assertEqual(self.turn(history, "8 (900) 000-00-00")["text"], consultant.QUESTIONS["company"])
        lead = self.turn(history, "ООО Тест")["lead"]
        self.assertEqual(lead["phone"], "+79000000000")
        self.assertEqual(lead["company"], "ООО Тест")
        self.assertEqual(len(lead["summary"].splitlines()), 3)
        self.assertIsNone(self.turn(history, "Спасибо")["lead"])

    def test_offer_refusal_and_explicit_restart(self):
        history = []
        with patch.multiple(consultant.solution, BASE_URL="test", API_KEY="test"), \
             patch.object(consultant, "_search", return_value=[("source.md", "Текст")]), \
             patch.object(consultant, "_generate", return_value=("Подойдёт АСКУЭ.", ["source.md"], True)):
            response = self.turn(history, "На заводе нужен удалённый сбор показаний.")
            self.assertTrue(response["text"].endswith(consultant.OFFER))
            self.assertEqual([s["step"] for s in response["trace"]], ["make_lead", "search", "llm"])
            self.assertEqual(self.turn(history, "Нет, спасибо")["text"], consultant.DECLINED)
            self.assertNotIn(consultant.OFFER, self.turn(history, "Расскажите подробнее")["text"])
        self.assertEqual(self.turn(history, "Хочу оставить заявку")["text"], consultant.QUESTIONS["name"])

    def test_policy_and_no_implicit_lead(self):
        for text in ("Назовите производителя панелей", "Дайте юридическое заключение"):
            response = self.turn([], text)
            self.assertEqual(response["trace"][-1]["step"], "policy")
            self.assertIsNone(response["lead"])
        self.assertIsNone(self.turn([], "+7 900 000 00 00")["lead"])
        self.assertEqual(self.turn([], "Не хочу оставлять заявку")["text"], consultant.DECLINED)

    def test_all_fields_and_summary(self):
        history = [{"role": "user", "content": "На заводе нужны показания"}]
        reply = self.turn(history, "Оформите заявку; ФИО: Иван Иванов; телефон: +7 900 000-00-00; компания: ООО Тест")
        self.assertIn("На заводе нужны показания", reply["lead"]["summary"])
        self.assertNotIn("900", reply["lead"]["summary"])

    def test_model_source_indexes(self):
        payload = {"choices": [{"message": {"content": '{"text":"Ответ", "sources":[1,1,0,3,true,2], "offer_lead":true}'}}]}
        with patch.object(consultant.solution, "post", return_value=payload):
            self.assertEqual(consultant._generate([], [("a", "x"), ("b", "y")]), ("Ответ", ["a", "b"], True))

    def test_missing_materials_and_model_failure(self):
        with patch.multiple(consultant.solution, BASE_URL="test", API_KEY="test"):
            with patch.object(consultant, "_search", return_value=[]):
                reply = self.turn([], "Какова точная цена без исходных данных?")
                self.assertIn("нет точного ответа", reply["text"])
                self.assertEqual(reply["sources"], [])
            with patch.object(consultant, "_search", return_value=[("a", "x")]), \
                 patch.object(consultant.solution, "post", return_value=None):
                self.assertEqual(self.turn([], "Как приготовить борщ?")["text"], consultant.UNAVAILABLE)
