from __future__ import annotations

import asyncio
import datetime
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

from app_config import get_app_config_value
from memory.memory_manager import load_memory, update_memory, delete_memory, format_memory_for_prompt
from sys_info import sys_info
from calendar_actions import get_calendar_events, add_calendar_event, delete_calendar_event
from reminders import get_reminders, add_reminder
from weather import get_weather_summary
from youtube_stats import get_youtube_channel_report
from tool_defs import TOOL_DECLARATIONS

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "core" / "prompt.txt"

app = FastAPI(title="Seymur's J.A.R.V.I.S. Cloud Core")

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-latest"

def get_api_key() -> str:
    return str(get_app_config_value("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", ""))

def load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "Sen Seymur'un JARVIS'isin — Bulutta çalışan kişisel AI asistanısın. "
            "Türkçe konuş. Kısa ve net yanıtlar ver. "
            "Araçları kullanarak görevleri tamamla."
        )

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Seymur's J.A.R.V.I.S.</title>
    <style>
        body {
            background-color: #030712;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            overflow: hidden;
        }
        h1 {
            color: #00f3ff;
            font-size: 24px;
            letter-spacing: 2px;
            margin-bottom: 5px;
        }
        #status {
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 40px;
        }
        .reactor {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: #0055ff;
            border: 3px solid #00f3ff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 0 30px rgba(0, 243, 255, 0.4);
            transition: all 0.3s ease;
        }
        .reactor.active {
            background: #00ff55;
            box-shadow: 0 0 40px rgba(0, 255, 85, 0.6);
        }
        .reactor span {
            font-weight: bold;
            font-size: 16px;
            letter-spacing: 1px;
        }
        #log {
            margin-top: 30px;
            width: 85%;
            max-width: 350px;
            height: 100px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            font-size: 12px;
            overflow-y: auto;
            color: #38bdf8;
        }
    </style>
</head>
<body>
    <h1>Seymur's J.A.R.V.I.S.</h1>
    <div id="status">Bağlantı kuruluyor...</div>
    
    <div class="reactor" id="reactor" onclick="toggleListening()">
        <span id="btnText">BAĞLAN</span>
    </div>

    <div id="log">Sistem hazır bekleniyor...</div>

    <script>
        let ws = null;
        let isListening = false;
        const statusEl = document.getElementById("status");
        const btnText = document.getElementById("btnText");
        const reactor = document.getElementById("reactor");
        const logEl = document.getElementById("log");

        function log(msg) {
            logEl.innerHTML += "<br>" + msg;
            logEl.scrollTop = logEl.scrollHeight;
        }

        function connect() {
            const proto = location.protocol === "https:" ? "wss://" : "ws://";
            ws = new WebSocket(proto + location.host + "/ws");

            ws.onopen = () => {
                statusEl.innerText = "Sistem Aktif - Hazır";
                btnText.innerText = "DİNLE";
                log("Bulut bağlantısı sağlandı.");
            };

            ws.onclose = () => {
                statusEl.innerText = "Bağlantı koptu, yeniden bağlanılıyor...";
                btnText.innerText = "YENİDEN";
                reactor.classList.remove("active");
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                log("JARVIS: " + event.data);
            };
        }

        function toggleListening() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            isListening = !isListening;
            if (isListening) {
                reactor.classList.add("active");
                statusEl.innerText = "Dinleniyor...";
                ws.send(JSON.stringify({type: "start_session"}));
            } else {
                reactor.classList.remove("active");
                statusEl.innerText = "Durduruldu.";
                ws.send(JSON.stringify({type: "stop_session"}));
            }
        }

        connect();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = genai.Client(api_key=get_api_key(), http_options={"api_version": "v1alpha"})
    
    memory = load_memory()
    mem_str = format_memory_for_prompt(memory)
    sys_p = load_system_prompt()
    now = datetime.datetime.now()
    time_ctx = f"[ŞU ANKİ ZAMAN]\n{now.strftime('%A, %d %B %Y — %H:%M')}\n\n"
    
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=time_ctx + (mem_str + "\n\n" if mem_str else "") + sys_p,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            while True:
                data = await websocket.receive_text()
                packet = json.loads(data)
                
                if packet.get("type") == "start_session":
                    await session.send_client_content(
                        turns={"parts": [{"text": "Sistem aktif, Seymur seni tetikledi."}]},
                        turn_complete=True
                    )
                    
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.text:
                                    await websocket.send_text(part.text)
                        
                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                name = fc.name
                                args = dict(fc.args or {})
                                result = "Tamamlandı."
                                
                                if name == "sys_info":
                                    result = sys_info(args.get("query", "all"))
                                elif name == "get_weather":
                                    result = get_weather_summary(args.get("location") or None)
                                elif name == "get_calendar_events":
                                    result = get_calendar_events(args.get("query", "today"), int(args.get("limit", 6) or 6))
                                elif name == "get_reminders":
                                    result = get_reminders(args.get("query", "upcoming"), int(args.get("limit", 8) or 8))
                                
                                await session.send_tool_response(
                                    function_responses=[types.FunctionResponse(id=fc.id, name=name, response={"result": result})]
                                )
                                await websocket.send_text(f"İşlem yapıldı: {name}")

    except WebSocketDisconnect:
        print("[JARVIS Cloud] Telefon bağlantısı kesildi.")
    except Exception as e:
        print(f"[JARVIS Cloud Error] {e}")
