import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from src.data_types import ChatHistoryPair, ChatMessageRecord
from src.db_service.postgres_controllers import PostgresChatService
from src.db_service.schemas import ChatExchange, ChatSession, SourceRecordTable
from src.main import record_cached_answer


class PostgresChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine: Engine = create_engine("sqlite:///:memory:")
        SourceRecordTable.__table__.create(self.engine)
        ChatSession.__table__.create(self.engine)
        ChatExchange.__table__.create(self.engine)
        self.chat_service = PostgresChatService(self.engine)
        with self.chat_service.session_factory() as session:
            session.add(
                SourceRecordTable(
                    record_id="record-1",
                    source_file="data/samples/convfinqa_dev_sample.json",
                    record_index=0,
                    split="dev",
                    has_type2_question=False,
                    has_duplicate_columns=False,
                    has_non_numeric_values=False,
                    num_dialogue_turns=1,
                )
            )
            session.commit()
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

        cached = self.chat_service.get_cached("What changed?", "record-1")

        self.assertIsInstance(cached, ChatMessageRecord)
        self.assertEqual(cached.role, "assistant")
        self.assertEqual(
            cached.content,
            '{"answer":"The linked assistant is returned.","citations":[]}',
        )

    def test_get_cached_ignores_invalid_assistant_answer(self) -> None:
        self._record_turn(
            prompt="What changed?",
            answer='{"answer":"This answer was invalidated.","citations":[]}',
        )
        with self.chat_service.session_factory() as session:
            assistant_row = session.execute(
                select(ChatExchange).where(ChatExchange.role == "assistant")
            ).scalar_one()
            assistant_row.invalid = True
            session.commit()

        cached = self.chat_service.get_cached("What changed?", "record-1")

        self.assertIsNone(cached)

    def test_cached_answer_is_recorded_as_new_history_pair(self) -> None:
        self._record_turn(
            prompt="What changed?",
            answer='{"answer":"The cached answer is replayed.","citations":[]}',
        )
        cached = self.chat_service.get_cached("What changed?", "record-1")
        self.assertIsNotNone(cached)

        response = record_cached_answer(
            self.chat_service,
            "What changed?",
            self.chat_session,
            cached,
            "record-1",
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.answer, "The cached answer is replayed.")
        updated_session = self.chat_service.get_session("record-1")
        self.assertIsNotNone(updated_session)
        self.assertEqual(updated_session.message_count, 4)
        self.assertEqual(updated_session.last_message_index, 3)
        history = self.chat_service.show_history(self.chat_session.session_id, limit=4)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].user_question.content, "What changed?")
        self.assertEqual(
            history[0].assistant.content,
            '{"answer":"The cached answer is replayed.","citations":[]}',
        )
        replayed_cached = self.chat_service.get_cached("What changed?", "record-1")
        self.assertIsNotNone(replayed_cached)
        self.assertEqual(
            replayed_cached.content,
            '{"answer":"The cached answer is replayed.","citations":[]}',
        )

    def test_get_cached_returns_none_for_unanswered_prompt(self) -> None:
        self.chat_service.record_user_message(
            self.chat_session.session_id,
            "No answer yet?",
        )

        self.assertIsNone(self.chat_service.get_cached("No answer yet?", "record-1"))

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

        cached = self.chat_service.get_cached("Repeatable?", "record-1")

        self.assertIsNotNone(cached)
        self.assertEqual(cached.role, "assistant")
        self.assertIn(
            cached.content,
            {
                '{"answer":"First answer.","citations":[]}',
                '{"answer":"Second answer.","citations":[]}',
            },
        )

    def test_get_session_returns_most_recent_session_for_record(self) -> None:
        second_session = self.chat_service.create_session("record-1")
        old_time = datetime(2026, 5, 5, 12, 0, tzinfo=UTC)
        new_time = old_time + timedelta(minutes=5)
        with self.chat_service.session_factory() as session:
            first_row = session.get(ChatSession, self.chat_session.session_id)
            second_row = session.get(ChatSession, second_session.session_id)
            self.assertIsNotNone(first_row)
            self.assertIsNotNone(second_row)
            first_row.last_message_at = old_time
            first_row.created_at = old_time
            second_row.last_message_at = new_time
            second_row.created_at = new_time
            session.commit()

        resumed = self.chat_service.get_session("record-1")

        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.session_id, second_session.session_id)

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
