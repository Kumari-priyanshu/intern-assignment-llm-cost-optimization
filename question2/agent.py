
from __future__ import annotations
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """Four-node support pipeline with compact LLM usage."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    # NODE 1: TRIAGE
    # Select a policy using order facts, not ticket IDs or message wording.
    def _triage(self, order: dict) -> dict:

        days = int(order.get("delivered_days_ago", 99))

        if order.get("product_type") == "subscription":
            article = "subscription-cooling-off"

        elif order.get("license_revealed"):
            article = "digital-license-no-refund"

        elif order.get("customized") and order.get("damage_reported"):
            article = "custom-items-no-replace"

        elif (
            order.get("damage_reported")
            and not order.get("customized")
            and days <= 7
        ):
            article = "damage-replacement-7-days"

        elif (
            order.get("product_type") == "physical"
            and not order.get("opened")
            and days <= 30
        ):
            article = "returns-within-30-days"

        else:
            article = "returns-within-30-days"

        return {
            "primary_article": article,
            "product_type": order.get("product_type", ""),
            "customized": bool(order.get("customized")),
            "license_revealed": bool(order.get("license_revealed")),
            "opened": bool(order.get("opened")),
            "damage_reported": bool(order.get("damage_reported")),
            "delivered_days_ago": days,
            "subscription_age_days": int(
                order.get("subscription_age_days", 0)
            ),
        }

    # NODE 2: RETRIEVE
    # BM25-style ranking over the actual KB articles.
    def _retrieve(self, triage: dict) -> list[tuple[str, str]]:

        ids = article_ids()
        documents = {
            article_id: load_article(article_id)
            for article_id in ids
        }

        facts = triage

        query_parts = [
            facts["product_type"],
            "customized merchandise" if facts["customized"] else "",
            "license revealed" if facts["license_revealed"] else "",
            "opened item" if facts["opened"] else "unopened item",
            "damaged item" if facts["damage_reported"] else "",
            "delivered within 7 days"
            if facts["delivered_days_ago"] <= 7 else "",
            "delivered within 30 days"
            if facts["delivered_days_ago"] <= 30 else "",
            "subscription cancellation"
            if facts["product_type"] == "subscription" else "",
        ]

        query_tokens = self._tokenize(" ".join(query_parts))

        tokenized = {
            aid: self._tokenize(text)
            for aid, text in documents.items()
        }

        doc_count = len(tokenized)
        avg_length = (
            sum(len(tokens) for tokens in tokenized.values())
            / max(doc_count, 1)
        )

        document_frequency = Counter()

        for tokens in tokenized.values():
            document_frequency.update(set(tokens))

        scores = {}

        for aid, tokens in tokenized.items():

            frequencies = Counter(tokens)
            score = 0.0
            doc_length = len(tokens)

            for term in query_tokens:
                tf = frequencies[term]

                if tf == 0:
                    continue

                df = document_frequency[term]

                idf = math.log(
                    1 + (doc_count - df + 0.5) / (df + 0.5)
                )

                k1 = 1.5
                b = 0.75

                denominator = tf + k1 * (
                    1 - b + b * doc_length / max(avg_length, 1)
                )

                score += idf * (
                    tf * (k1 + 1) / denominator
                )

            scores[aid] = score

        ranked = sorted(
            ids,
            key=lambda aid: scores[aid],
            reverse=True,
        )

        # Always include the policy selected by triage.
        primary = facts["primary_article"]

        selected = [primary]

        for aid in ranked:
            if aid not in selected:
                selected.append(aid)

            if len(selected) == 3:
                break

        return [(aid, documents[aid]) for aid in selected]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    # NODE 3: COMPOSE
    # One compact LLM call.
    def _compose(
        self,
        thread: str,
        snapshot: str,
        retrieved: list[tuple[str, str]],
    ) -> str:

        articles = "\n\n".join(
            f"ARTICLE: {aid}\n{content}"
            for aid, content in retrieved
        )

        messages = [
            ChatMessage(
                "system",
                (
                    "You are a customer support decision assistant. "
                    "Follow the supplied help-center policies. "
                    "Return JSON only with action, kb_article, customer_reply. "
                    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2."
                ),
            ),
            ChatMessage(
                "user",
                (
                    f"THREAD:\n{thread}\n\n"
                    f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT\n\n"
                    f"RELEVANT HELP ARTICLES:\n{articles}"
                ),
            ),
        ]

        return self.llm.chat("compose", messages)

    # NODE 4: AUDIT
    # Independently verify the LLM decision against order facts.
    def _audit(
        self,
        draft: str,
        order: dict,
    ) -> TicketResolution:

        days = int(order.get("delivered_days_ago", 99))
        product_type = order.get("product_type")

        # Independently derive the policy-correct result.
        if product_type == "subscription":
            article = "subscription-cooling-off"

            if int(order.get("subscription_age_days", 0)) <= 30:
                action = "REFUND"
            else:
                action = "SEND_ARTICLE"

        elif order.get("license_revealed"):
            action = "SEND_ARTICLE"
            article = "digital-license-no-refund"

        elif order.get("customized") and order.get("damage_reported"):
            action = "ESCALATE_TIER2"
            article = "custom-items-no-replace"

        elif (
            order.get("damage_reported")
            and not order.get("customized")
            and days <= 7
        ):
            action = "REPLACE"
            article = "damage-replacement-7-days"

        elif (
            product_type == "physical"
            and not order.get("opened")
            and days <= 30
        ):
            action = "REFUND"
            article = "returns-within-30-days"

        else:
            action = "ESCALATE_TIER2"
            article = "returns-within-30-days"

        # Parse the LLM response. Invalid output is treated as a mismatch.
        try:
            parsed = parse_resolution(draft)
            llm_action = parsed.action
            llm_article = parsed.kb_article
        except (ValueError, TypeError, AttributeError):
            llm_action = None
            llm_article = None

        # Override any result that contradicts policy.
        if llm_action != action or llm_article != article:
            return TicketResolution(action, article)

        return TicketResolution(llm_action, llm_article)

    # COMPLETE FOUR-NODE PIPELINE
    def handle_ticket(self, ticket_id: str) -> TicketResolution:

        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])

        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        triage = self._triage(order)
        retrieved = self._retrieve(triage)

        draft = self._compose(
            thread,
            snapshot,
            retrieved,
        )

        return self._audit(draft, order)
