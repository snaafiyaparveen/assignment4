import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama import ResponseError
from openai import APIConnectionError, APIError, AuthenticationError, NotFoundError

load_dotenv()

SYSTEM_PROMPT = """You are a helpful, empathetic e-commerce customer support assistant.
Answer questions about delayed orders, damaged products, returns, and refunds.

Write in a warm, calm, natural tone. Acknowledge the customer's concern briefly,
then answer their question directly. Keep replies concise: use a short paragraph
for simple questions, or a numbered list of up to three practical next steps when
the customer needs to take action. Use plain language and avoid repetitive
greetings, filler, excessive apologies, and unnecessary headings. Ask one focused
follow-up question only when the answer depends on missing information.

Use the conversation for context, but do not invent order details, store policies,
delivery estimates, refund status, or actions you have taken. Do not present
specific waiting periods or procedures as store policy unless they are provided
in the conversation. You are not connected to a store: do not say "we", "our",
or imply you can investigate or resolve an order. Do not ask the customer for an
order number, tracking number, address, or other identifying details; direct them
to share relevant details only through the retailer's official support channel.
If a policy or account detail is unknown, say so plainly and point the customer
to that channel. Never claim that a return, refund, replacement, or order change
has been completed. Do not ask for passwords, payment-card details, or other
sensitive credentials. End with the next useful action instead of an unnecessary
follow-up question."""

QUICK_QUESTIONS = (
    {
        "title": "A delayed order",
        "icon": ":material/local_shipping:",
        "description": "Find practical next steps when a delivery is late.",
        "question": "My order is delayed. What should I do?",
    },
    {
        "title": "A damaged item",
        "icon": ":material/inventory_2:",
        "description": "Learn what to document and how to get help.",
        "question": "My item arrived damaged. What are my next steps?",
    },
    {
        "title": "A return",
        "icon": ":material/keyboard_return:",
        "description": "Understand what to check before returning an item.",
        "question": "I'd like to return an item. What should I check first?",
    },
    {
        "title": "A refund",
        "icon": ":material/payments:",
        "description": "Get guidance if a refund hasn't arrived.",
        "question": "My refund hasn't arrived yet. What should I do?",
    },
)

FOLLOW_UP_QUESTIONS = (
    "What information should I have ready?",
    "How do I contact the store?",
    "What if that doesn't work?",
)

PROVIDER_ERROR_MESSAGES = {
    "openrouter_auth": "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in your .env file.",
    "gemini_auth": "Google Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file.",
    "gemini_model": "The Gemini model was not found. Check GEMINI_MODEL and confirm it is available to your API key.",
    "gemini_unavailable": "Google Gemini is temporarily unavailable. Check your connection and try again.",
    "gemini_request": "Google Gemini could not complete the request. Check model access and API availability.",
    "model_not_found": "The selected model was not found. Check the model name and provider availability.",
    "openrouter_connection": "Could not connect to OpenRouter. Check your internet connection and try again.",
    "openrouter_request": "OpenRouter could not complete the request. Check your model, account, and API key.",
    "ollama": "Ollama could not load or run this model. Check that Ollama is running and the model is installed.",
    "provider_connection": "Could not connect to the selected model provider. Check that the service is available and try again.",
    "empty_response": "The model returned an empty response. Try asking again.",
}


def queue_question(question):
    st.session_state["pending_question"] = question


def clear_conversation():
    st.session_state.messages = []
    st.session_state["pending_question"] = None
    st.session_state["failed_question"] = None
    st.session_state["failed_error_code"] = None
    st.session_state["failed_error_message"] = None
    st.session_state["can_retry_question"] = False
    st.session_state["retrying_question"] = False


def mark_failed_question(question, error_code, retryable=True):
    st.session_state["failed_question"] = question
    st.session_state["failed_error_code"] = error_code
    st.session_state["failed_error_message"] = None
    st.session_state["can_retry_question"] = retryable


def retry_failed_question():
    failed_question = st.session_state.get("failed_question")
    if failed_question and st.session_state.get("can_retry_question"):
        st.session_state["pending_question"] = failed_question
        st.session_state["retrying_question"] = True
        st.session_state["failed_question"] = None
        st.session_state["failed_error_code"] = None
        st.session_state["failed_error_message"] = None
        st.session_state["can_retry_question"] = False


def show_provider_error(error_code, error_message=None):
    if error_code == "gemini_quota":
        with st.container(border=True):
            st.warning(
                "The assistant has reached this API key's Gemini request limit.",
                icon=":material/hourglass_top:",
            )
            st.caption(
                "Your message is saved in this conversation, but Gemini "
                "can't generate a reply until the limit resets."
            )
            st.link_button(
                "Check usage and limits",
                "https://ai.google.dev/gemini-api/docs/rate-limits",
                icon=":material/open_in_new:",
            )
            with st.expander("What can I do?"):
                st.markdown(
                    "Wait for Google's quota to reset, enable billing for "
                    "the API project, or configure another provider in `.env` "
                    "and restart the app."
                )
    else:
        st.error(
            error_message
            or PROVIDER_ERROR_MESSAGES.get(
                error_code, "The model provider could not complete the request."
            )
        )


def completed_turn_history(messages):
    history = []
    for index in range(len(messages) - 1):
        current, following = messages[index : index + 2]
        if current["role"] == "user" and following["role"] == "assistant":
            history.extend(
                [
                    HumanMessage(content=current["content"]),
                    AIMessage(content=following["content"]),
                ]
            )
    return history


def build_support_chain():
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    model_settings = {
        "ollama": ("OLLAMA_MODEL", "qwen2.5:3b"),
        "openrouter": ("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "gemini": ("GEMINI_MODEL", "gemini-3.5-flash"),
    }
    if provider not in model_settings:
        raise ValueError(
            "LLM_PROVIDER must be 'ollama', 'openrouter', or 'gemini'."
        )

    model_env, default_model = model_settings[provider]
    model_name = os.getenv(model_env, default_model).strip()

    if not model_name:
        raise ValueError("The configured model name is empty.")

    if provider == "ollama":
        model = ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        )
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Add it to your .env file."
            )
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )
    else:
        raise ValueError("Unsupported LLM provider.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | model, provider, model_name


def response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            block["text"].strip()
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        )

    return ""


st.set_page_config(
    page_title="Customer support assistant",
    page_icon=":material/support_agent:",
)

with st.container(border=True):
    st.title("Customer support", icon=":material/support_agent:")
    st.caption("A little help with orders, returns, and refunds.")
    st.caption(
        "Get practical next steps. Your store's team is the best source for "
        "order-specific updates."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    support_chain, provider, model_name = build_support_chain()
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.title("Help desk", icon=":material/support_agent:")
    st.caption("Your shopping support, in one place.")
    st.badge(provider.title(), icon=":material/neurology:", color="blue")
    with st.expander("About this assistant", icon=":material/info:"):
        st.caption(f"Powered by {model_name}.")
        st.caption(
            "It can offer general guidance but cannot see your orders or store policies."
        )
        st.caption("Your conversation stays in this browser session.")
    st.button(
        "Start a new chat",
        icon=":material/add_comment:",
        width="stretch",
        on_click=clear_conversation,
    )

retrying_question = st.session_state.pop("retrying_question", False)
typed_question = st.chat_input(
    "Ask about an order, return, or refund…",
    key="customer_question",
)
question = st.session_state.pop("pending_question", None) or typed_question

if not st.session_state.messages and not question:
    st.subheader("What can we help with?")
    st.caption("Pick a topic to start a conversation, or ask in your own words.")
    for row_start in (0, 2):
        topic_columns = st.columns(2, gap="medium")
        for column, topic in zip(
            topic_columns, QUICK_QUESTIONS[row_start : row_start + 2]
        ):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {topic['icon']} {topic['title']}")
                    st.caption(topic["description"])
                    st.button(
                        "Explore this topic",
                        key=f"topic_{row_start}_{topic['title']}",
                        icon=":material/arrow_forward:",
                        width="stretch",
                        on_click=queue_question,
                        args=(topic["question"],),
                    )

visible_messages = (
    st.session_state.messages[:-1]
    if retrying_question
    else st.session_state.messages
)
for message in visible_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    if retrying_question:
        if (
            not st.session_state.messages
            or st.session_state.messages[-1]["role"] != "user"
            or st.session_state.messages[-1]["content"] != question
        ):
            raise ValueError("The question selected for retry is no longer available.")
        with st.chat_message("user"):
            st.markdown(question)
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    history = completed_turn_history(st.session_state.messages[:-1])

    error_code = None
    try:
        response = support_chain.invoke(
            {"history": history, "question": question}
        )
        answer = response_text(response.content)
        if not answer:
            raise ValueError("The model returned an empty response.")
    except GoogleRateLimitError:
        error_code = "gemini_quota"
        mark_failed_question(question, error_code, retryable=False)
    except AuthenticationError:
        error_code = "openrouter_auth"
    except GoogleAuthenticationError:
        error_code = "gemini_auth"
    except GoogleModelNotFoundError:
        error_code = "gemini_model"
    except GoogleAPIError:
        error_code = "gemini_unavailable"
    except ChatGoogleGenerativeAIError:
        error_code = "gemini_request"
    except NotFoundError:
        error_code = "model_not_found"
    except APIConnectionError:
        error_code = "openrouter_connection"
    except APIError:
        error_code = "openrouter_request"
    except ResponseError:
        error_code = "ollama"
    except httpx.HTTPError:
        error_code = "provider_connection"
    except ValueError as error:
        if str(error) == "The model returned an empty response.":
            error_code = "empty_response"
        else:
            st.session_state["failed_question"] = question
            st.session_state["failed_error_code"] = "request_error"
            st.session_state["failed_error_message"] = str(error)
            st.session_state["can_retry_question"] = False
    else:
        st.session_state["failed_question"] = None
        st.session_state["failed_error_code"] = None
        st.session_state["failed_error_message"] = None
        st.session_state["can_retry_question"] = False
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    if error_code:
        mark_failed_question(
            question,
            error_code,
            retryable=error_code != "gemini_quota",
        )

if (
    not question
    and st.session_state.get("failed_question")
    and st.session_state.get("failed_error_code")
):
    with st.chat_message("assistant"):
        show_provider_error(
            st.session_state["failed_error_code"],
            st.session_state.get("failed_error_message"),
        )

if (
    question
    and st.session_state.get("failed_question") == question
    and st.session_state.get("failed_error_code")
):
    with st.chat_message("assistant"):
        show_provider_error(
            st.session_state["failed_error_code"],
            st.session_state.get("failed_error_message"),
        )

if (
    question
    and not st.session_state.get("failed_question")
    and st.session_state.messages[-1]["role"] == "assistant"
):
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        st.markdown(st.session_state.messages[-1]["content"])

if (
    st.session_state.get("failed_question")
    and st.session_state.get("can_retry_question")
):
    st.button(
        "Retry last question",
        icon=":material/refresh:",
        on_click=retry_failed_question,
    )

if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
):
    st.caption("Need help with something else?")
    with st.container(horizontal=True):
        for index, follow_up in enumerate(FOLLOW_UP_QUESTIONS):
            st.button(
                follow_up,
                key=f"follow_up_{index}",
                on_click=queue_question,
                args=(follow_up,),
                icon=":material/arrow_forward:",
            )
