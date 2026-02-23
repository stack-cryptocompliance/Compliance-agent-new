from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from supabase import create_client
import os
import openai
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI()

# Enable CORS so Vercel frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility: stream OpenAI responses
async def stream_openai(messages):
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    async for event in response:
        if event.choices[0].delta.get("content"):
            yield event.choices[0].delta.content

# Endpoint: chat
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "user123")
    question = data.get("question", "")

    # 1️⃣ Retrieve last 10 messages from Supabase for conversation memory
    memory_res = supabase.table("conversations") \
        .select("role, content") \
        .eq("user_id", user_id) \
        .order("created_at", desc=False) \
        .limit(10) \
        .execute()

    memory_messages = memory_res.data if memory_res.data else []

    # Convert to OpenAI message format
    messages = [{"role": m["role"], "content": m["content"]} for m in memory_messages]
    messages.append({"role": "user", "content": question})

    # 2️⃣ Save user message to Supabase
    supabase.table("conversations").insert({
        "user_id": user_id,
        "role": "user",
        "content": question
    }).execute()

    # 3️⃣ Stream AI response
    async def event_generator():
        async for chunk in stream_openai(messages):
            yield chunk

    # 4️⃣ Save AI response to Supabase after generating (non-blocking)
    async def save_response(response_text):
        supabase.table("conversations").insert({
            "user_id": user_id,
            "role": "assistant",
            "content": response_text
        }).execute()

    # Collect chunks for saving
    collected = []
    async for chunk in event_generator():
        collected.append(chunk)
        yield chunk

    response_text = "".join(collected)
    await save_response(response_text)

    return StreamingResponse(event_generator(), media_type="text/plain")