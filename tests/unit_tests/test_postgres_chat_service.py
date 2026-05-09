import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from src.data_types import CachedAnswerRecord, ChatHistoryPair, ChatMessageRecord
from src.db_service.postgres_controllers import PostgresChatService
from src.db_service.schemas import (
    AnswerCache,
    ChatExchange,
    ChatSession,
    SourceRecordTable,
)
from src.main import process_question, record_cached_answer
from src.rag_service import RagAnswer


class FakeAnswerService:
    def __init__(self, answers: list[RagAnswer]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, list[ChatHistoryPair], bool]] = []

    def answer(
        self,
        message: str,
        record_id: str,
        session_history: list[ChatHistoryPair] | None = None,
        is_requery: bool = False,
    ) -> RagAnswer:
        self.calls.append((message, record_id, session_history or [], is_requery))
        return self.answers.pop(0)


def rag_answer_json(answer: str, citations: list[str] | None = None) -> str:
    citation_text = ",".join(f'"{citation}"' for citation in citations or [])
    return (
        f'{{"answer":"{answer}","citations":[{citation_text}],'
        f'"calculation_program":null,"turn_program":null,"requery":null}}'
    )


class PostgresChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine: Engine = create_engine("sqlite:///:memory:")
        SourceRecordTable.__table__.create(self.engine)
        ChatSession.__table__.create(self.engine)
        ChatExchange.__table__.create(self.engine)
        AnswerCache.__table__.create(self.engine)
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
        answer: str = rag_answer_json("The cache works."),
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

    def _record_cached_turn(
        self,
        prompt: str = "What changed?",
        answer: str = rag_answer_json("The cache works."),
    ) -> tuple[ChatMessageRecord, None]:
        turn = self._record_turn(prompt, answer)
        self.chat_service.cache_answer(prompt, "record-1", answer)
        return turn

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
            cached_rows = session.execute(select(AnswerCache)).scalars().all()
            self.assertEqual(cached_rows, [])

    def test_get_cached_returns_cached_answer(self) -> None:
        self._record_cached_turn(
            prompt="What changed?",
            answer=rag_answer_json("The cached assistant is returned."),
        )

        cached = self.chat_service.get_cached("What changed?", "record-1")

        self.assertIsInstance(cached, CachedAnswerRecord)
        self.assertEqual(
            cached.content,
            rag_answer_json("The cached assistant is returned."),
        )

        with self.chat_service.session_factory() as session:
            self.assertEqual(
                session.execute(select(AnswerCache)).scalar_one().prompt,
                "What changed?",
            )

    def test_get_cached_ignores_invalid_assistant_answer(self) -> None:
        self._record_cached_turn(
            prompt="What changed?",
            answer=rag_answer_json("This answer was invalidated."),
        )
        with self.chat_service.session_factory() as session:
            cache_row = session.execute(select(AnswerCache)).scalar_one()
            cache_row.invalid = True
            session.commit()

        cached = self.chat_service.get_cached("What changed?", "record-1")

        self.assertIsNone(cached)

    def test_cached_answer_is_recorded_as_new_history_pair(self) -> None:
        self._record_cached_turn(
            prompt="What changed?",
            answer=rag_answer_json("The cached answer is replayed."),
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
            rag_answer_json("The cached answer is replayed."),
        )
        replayed_cached = self.chat_service.get_cached("What changed?", "record-1")
        self.assertIsNotNone(replayed_cached)
        self.assertEqual(
            replayed_cached.content,
            rag_answer_json("The cached answer is replayed."),
        )
        with self.chat_service.session_factory() as session:
            cache_count = session.execute(select(AnswerCache)).scalars().all()
            self.assertEqual(len(cache_count), 1)

    def test_process_question_cache_hit_records_one_new_history_pair(self) -> None:
        self._record_cached_turn(
            prompt="What changed?",
            answer=rag_answer_json("The cached answer is replayed."),
        )
        answer_service = FakeAnswerService([])

        response = process_question(
            "What changed?",
            "record-1",
            self.chat_session,
            SimpleNamespace(caching=True),
            self.chat_service,
            answer_service,
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.answer, "The cached answer is replayed.")
        self.assertEqual(answer_service.calls, [])
        updated_session = self.chat_service.get_session("record-1")
        self.assertIsNotNone(updated_session)
        self.assertEqual(updated_session.message_count, 4)
        history = self.chat_service.show_history(self.chat_session.session_id, limit=4)
        self.assertEqual(len(history), 2)
        with self.chat_service.session_factory() as session:
            cache_count = session.execute(select(AnswerCache)).scalars().all()
            self.assertEqual(len(cache_count), 1)

    def test_process_question_cache_miss_records_cache_in_process_question(
        self,
    ) -> None:
        answer_service = FakeAnswerService(
            [RagAnswer(answer="fresh", citations=["chunk-1"])]
        )

        response = process_question(
            "What changed?",
            "record-1",
            self.chat_session,
            SimpleNamespace(caching=True),
            self.chat_service,
            answer_service,
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.answer, "fresh")
        cached = self.chat_service.get_cached("What changed?", "record-1")
        self.assertIsNotNone(cached)
        self.assertEqual(
            RagAnswer.model_validate_json(cached.content).answer,
            "fresh",
        )

    def test_process_question_requery_records_only_final_history_pair(self) -> None:
        answer_service = FakeAnswerService(
            [
                RagAnswer(
                    answer="I don't know",
                    citations=[],
                    requery="net income 2008",
                ),
                RagAnswer(answer="123", citations=["chunk-1"]),
            ]
        )

        response = process_question(
            "what 2008",
            "record-1",
            self.chat_session,
            SimpleNamespace(caching=False),
            self.chat_service,
            answer_service,
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.answer, "123")
        self.assertEqual(response.requery, "net income 2008")
        self.assertEqual(
            [call[0] for call in answer_service.calls],
            ["what 2008", "net income 2008"],
        )
        self.assertEqual(
            [call[3] for call in answer_service.calls],
            [False, True],
        )
        updated_session = self.chat_service.get_session("record-1")
        self.assertIsNotNone(updated_session)
        self.assertEqual(updated_session.message_count, 2)
        history = self.chat_service.show_history(self.chat_session.session_id, limit=2)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].user_question.content, "what 2008")
        stored_answer = RagAnswer.model_validate_json(history[0].assistant.content)
        self.assertEqual(stored_answer.answer, "123")
        self.assertEqual(stored_answer.requery, "net income 2008")

    def test_get_cached_returns_none_for_unanswered_prompt(self) -> None:
        self.chat_service.record_user_message(
            self.chat_session.session_id,
            "No answer yet?",
        )

        self.assertIsNone(self.chat_service.get_cached("No answer yet?", "record-1"))

    def test_get_cached_handles_duplicate_prompts_without_multiple_rows_error(
        self,
    ) -> None:
        self._record_cached_turn(
            prompt="Repeatable?",
            answer=rag_answer_json("First answer."),
        )
        self._record_cached_turn(
            prompt="Repeatable?",
            answer=rag_answer_json("Second answer."),
        )

        cached = self.chat_service.get_cached("Repeatable?", "record-1")

        self.assertIsNotNone(cached)
        self.assertIn(
            cached.content,
            {
                rag_answer_json("First answer."),
                rag_answer_json("Second answer."),
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
            answer=rag_answer_json("Answer one."),
        )

        history = self.chat_service.show_history(self.chat_session.session_id, limit=2)

        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], ChatHistoryPair)
        self.assertEqual(history[0].user_question.role, "user")
        self.assertEqual(history[0].user_question.content, "Question one?")
        self.assertEqual(history[0].assistant.role, "assistant")
        self.assertEqual(
            history[0].assistant.content,
            rag_answer_json("Answer one."),
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
