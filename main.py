from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from supabase import create_client
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Stream OpenAI response
async def stream_openai(messages):

    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )

    async for event in response:

        if event.choices[0].delta.get("content"):

            yield event.choices[0].delta.content



# ✅ Chat endpoint with RAG
@app.post("/chat")
async def chat(request: Request):

    data = await request.json()

    user_id = data.get("user_id", "user123")

    question = data.get("question", "")


    # =====================================================
    # ✅ STEP 1: CREATE EMBEDDING FOR RAG
    # =====================================================

    embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = embedding_response.data[0].embedding


    # =====================================================
    # ✅ STEP 2: SEARCH SUPABASE DOCUMENTS TABLE
    # =====================================================

    docs = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()


    rag_context = ""

    if docs.data:

        rag_context = "\n\n".join(
            doc["content"] for doc in docs.data
        )


    # =====================================================
    # ✅ STEP 3: GET MEMORY FROM conversations table
    # =====================================================

    memory_res = supabase.table("conversations") \
        .select("role, content") \
        .eq("user_id", user_id) \
        .order("created_at", desc=False) \
        .limit(10) \
        .execute()


    memory_messages = memory_res.data if memory_res.data else []


    # =====================================================
    # ✅ STEP 4: BUILD FINAL PROMPT
    # =====================================================

    messages = []


    # Inject RAG context

    messages.append({

        "role": "system",

        "content": f"""
You are a crypto compliance assistant.

Use this context to answer:

{rag_context}

If answer is not in context, say you don't know.
"""
    })


    # Add memory

    messages.extend([
        {
            "role": m["role"],
            "content": m["content"]
        }
        for m in memory_messages
    ])


    # Add user question

    messages.append({

        "role": "user",

        "content": question

    })

    supabase.table("conversations").insert({

        "user_id": user_id,

        "role": "user",

        "content": question

    }).execute()



    # =====================================================
    # ✅ STEP 6: STREAM RESPONSE + SAVE AI RESPONSE
    # =====================================================

    async def generator():

        collected = []

        async for chunk in stream_openai(messages):

            collected.append(chunk)

            yield chunk


        full_response = "".join(collected)


        supabase.table("conversations").insert({

            "user_id": user_id,

            "role": "assistant",

            "content": full_response

        }).execute()



    return StreamingResponse(generator(), media_type="text/plain")