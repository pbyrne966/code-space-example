import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from src.db_service.data_types import ChatHistoryPair, ChatMessageRecord
from src.db_service.postgres_controllers import PostgresChatService
from src.db_service.schemas import ChatExchange, ChatSession


class PostgresChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine: Engine = create_engine("sqlite:///:memory:")
        ChatSession.__table__.create(self.engine)
        ChatExchange.__table__.create(self.engine)
        self.chat_service = PostgresChatService(self.engine)
        self.chat_session = self.chat_service.create_session("record-1")

    def _record_turn(
        self,
        prompt: str = "What changed?",
        answer: str = '{"answer":"The cache works.","citations":[]}',
    ) -> tuple[ChatMessageRecord, None]:
        user_message = self.chat_service.record_user_message(
            self.chat_session.session_id,
            prompt,
        )
        assistant_message = self.chat_service.record_assistant_message(
            self.chat_session.session_id,
            answer,
            user_message,
        )
        return user_message, assistant_message

    def test_record_user_message_persists_and_returns_message_record(self) -> None:
        message = self.chat_service.record_user_message(
            self.chat_session.session_id,
            "What changed?",
        )

        self.assertIsInstance(message, ChatMessageRecord)
        self.assertIsNotNone(message.message_id)
        self.assertEqual(message.session_id, self.chat_session.session_id)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "What changed?")

        updated_session = self.chat_service.get_session("record-1")
        self.assertIsNotNone(updated_session)
        self.assertEqual(updated_session.message_count, 1)
        self.assertEqual(updated_session.last_message_index, 0)

    def test_record_assistant_message_stores_linked_user_message_id(self) -> None:
        user_message, _ = self._record_turn()

        with self.chat_service.session_factory() as session:
            assistant_row = session.execute(
                select(ChatExchange).where(ChatExchange.role == "assistant")
            ).scalar_one()
            self.assertEqual(assistant_row.linked_message_id, user_message.message_id)
            self.assertEqual(
                assistant_row.linked_message.message_id,
                user_message.message_id,
            )

    def test_get_cached_returns_linked_assistant_answer(self) -> None:
        self._record_turn(
            prompt="What changed?",
            answer='{"answer":"The linked assistant is returned.","citations":[]}',
        )

        cached = self.chat_service.get_cached("What changed?")

        self.assertIsInstance(cached, ChatMessageRecord)
        self.assertEqual(cached.role, "assistant")
        self.assertEqual(
            cached.content,
            '{"answer":"The linked assistant is returned.","citations":[]}',
        )

    def test_get_cached_returns_none_for_unanswered_prompt(self) -> None:
        self.chat_service.record_user_message(
            self.chat_session.session_id,
            "No answer yet?",
        )

        self.assertIsNone(self.chat_service.get_cached("No answer yet?"))

    def test_get_cached_handles_duplicate_prompts_without_multiple_rows_error(
        self,
    ) -> None:
        self._record_turn(
            prompt="Repeatable?",
            answer='{"answer":"First answer.","citations":[]}',
        )
        self._record_turn(
            prompt="Repeatable?",
            answer='{"answer":"Second answer.","citations":[]}',
        )

        cached = self.chat_service.get_cached("Repeatable?")

        self.assertIsNotNone(cached)
        self.assertEqual(cached.role, "assistant")
        self.assertIn(
            cached.content,
            {
                '{"answer":"First answer.","citations":[]}',
                '{"answer":"Second answer.","citations":[]}',
            },
        )

    def test_show_history_returns_linked_user_assistant_pairs(self) -> None:
        self._record_turn(
            prompt="Question one?",
            answer='{"answer":"Answer one.","citations":[]}',
        )

        history = self.chat_service.show_history(self.chat_session.session_id, limit=2)

        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], ChatHistoryPair)
        self.assertEqual(history[0].user_question.role, "user")
        self.assertEqual(history[0].user_question.content, "Question one?")
        self.assertEqual(history[0].assistant.role, "assistant")
        self.assertEqual(
            history[0].assistant.content,
            '{"answer":"Answer one.","citations":[]}',
        )

    def test_show_history_ignores_unpaired_user_messages(self) -> None:
        self.chat_service.record_user_message(
            self.chat_session.session_id,
            "No pair?",
        )

        self.assertEqual(
            self.chat_service.show_history(self.chat_session.session_id, limit=2),
            [],
        )


if __name__ == "__main__":
    unittest.main()
