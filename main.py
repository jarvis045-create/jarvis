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

app = FastAPI(title="J.A.R.V.I.S. Cloud Core")

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-latest"

def get_api_key() -> str:
    return str(get_app_config_value("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", ""))

def load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "Sen JARVIS'sin — Bulutta çalışan kişisel AI asistanısın. "
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
        :root {
            --c-bg: #030712;
            --c-pri: #00f0ff;
            --c-mid: #005f73;
            --c-text: #e0fbfc;
            --c-green: #06d6a0;
            --c-blue: #3a86ff;
            --c-gold: #ffd166;
            --c-red: #ef476f;
        }

        body {
            background-color: var(--c-bg);
            color: var(--c-text);
            font-family: 'Courier New', Courier, monospace;
            margin: 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            height: 94vh;
            box-sizing: border-box;
            overflow: hidden;
        }

        .header {
            text-align: center;
            border-bottom: 1px solid var(--c-mid);
            padding-bottom: 10px;
            margin-bottom: 10px;
        }

        .header h2 {
            margin: 0;
            color: var(--c-pri);
            font-size: 22px;
            letter-spacing: 2px;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
        }

        .header .status-sub {
            font-size: 11px;
            color: var(--c-mid);
            margin-top: 4px;
            letter-spacing: 1px;
        }

        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        /* ARC REACTOR BUTTON */
        .arc-reactor {
            width: 180px;
            height: 180px;
            border: 2px dashed var(--c-pri);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            animation: spin 15s linear infinite;
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
            margin: 15px 0;
            cursor: pointer;
            background: rgba(0, 240, 255, 0.03);
        }

        .arc-reactor::before {
            content: '';
            position: absolute;
            width: 140px;
            height: 140px;
            border: 1px solid var(--c-mid);
            border-radius: 50%;
            animation: spin-reverse 10s linear infinite;
        }

        .inner-core {
            width: 90px;
            height: 90px;
            border: 2px solid var(--c-pri);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.5);
        }

        .core-dot {
            width: 60px;
            height: 60px;
            background-color: var(--c-pri);
            border-radius: 50%;
            box-shadow: 0 0 20px var(--c-pri);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--c-bg);
            font-weight: bold;
            font-size: 11px;
            transition: all 0.3s ease;
            text-align: center;
        }

        .core-dot.listening { background-color: var(--c-green); box-shadow: 0 0 30px var(--c-green); }
        .core-dot.speaking { background-color: var(--c-blue); box-shadow: 0 0 30px var(--c-blue); }
        .core-dot.thinking { background-color: var(--c-gold); box-shadow: 0 0 30px var(--c-gold); }
        .core-dot.error { background-color: var(--c-red); box-shadow: 0 0 30px var(--c-red); }

        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes spin-reverse { 100% { transform: rotate(-360deg); } }

        .chat-box {
            width: 100%;
            height: 120px;
            background: rgba(2, 6, 23, 0.9);
            border: 1px solid var(--c-mid);
            border-radius: 6px;
            padding: 10px;
            overflow-y: auto;
            font-size: 11px;
            margin-bottom: 12px;
            box-sizing: border-box;
            color: #38bdf8;
        }

        .chat-box div {
            margin-bottom: 6px;
            line-height: 1.3;
        }

        .controls {
            display: flex;
            gap: 8px;
            width: 100%;
        }

        input {
            flex: 1;
            background: #020617;
            border: 1px solid var(--c-mid);
            color: var(--c-text);
            padding: 10px;
            border-radius: 4px;
            font-family: inherit;
            font-size: 13px;
        }

        input:focus {
            border-color: var(--c-pri);
            outline: none;
            box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
        }

        button {
            background: var(--c-mid);
            color: var(--c-text);
            border: 1px solid var(--c-pri);
            padding: 10px 14px;
            font-weight: bold;
            border-radius: 4px;
            cursor: pointer;
            font-family: inherit;
        }

        button:active {
            background: var(--c-pri);
            color: var(--c-bg);
        }
    </style>
</head>
<body>

    <div class="header">
        <h2>Seymur's J.A.R.V.I.S.</h2>
        <div class="status-sub" id="status-text">Bağlantı kuruluyor...</div>
    </div>

    <div class="main-container">
        <!-- Reaktor (Tıklayınca Oturum Başlatır/Durdurur) -->
        <div class="arc-reactor" onclick="toggleListening()">
            <div class="inner-core">
                <div class="core-dot" id="core-dot">BAĞLAN</div>
            </div>
        </div>

        <div class="chat-box" id="log">
            <div>Sistem hazır bekleniyor...</div>
        </div>
    </div>

    <div class="controls">
        <input type="text" id="cmd-input" placeholder="Komut yazın..." onkeypress="checkEnter(event)">
        <button onclick="sendCmdText()">GÖNDER</button>
    </div>

    <script>
        let ws = null;
        let isListening = false;
        const statusEl = document.getElementById("status-text");
        const coreDot = document.getElementById("core-dot");
        const logEl = document.getElementById("log");
        const cmdInput = document.getElementById("cmd-input");

        function log(msg, type="pri") {
            let col = "#38bdf8";
            if(type === "sys") col = "#ffd166";
            if(type === "you") col = "#00f0ff";
            if(type === "err") col = "#ef476f";
            
            logEl.innerHTML += `<div style="color:${col}">` + msg + `</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }

        function setCoreState(state) {
            coreDot.className = "core-dot " + state.toLowerCase();
            if(state === "LISTENING") coreDot.innerText = "DİNLİYOR";
            else if(state === "THINKING") coreDot.innerText = "DÜŞÜN";
            else if(state === "SPEAKING") coreDot.innerText = "KONUŞ";
            else if(state === "ERROR") coreDot.innerText = "HATA";
            else if(state === "ONLINE") coreDot.innerText = "DİNLE";
            else coreDot.innerText = "BAĞLAN";
            
            statusEl.innerText = "SİSTEM // " + state;
        }

        function connect() {
            const proto = location.protocol === "https:" ? "wss://" : "ws://";
            ws = new WebSocket(proto + location.host + "/ws");

            ws.onopen = () => {
                statusEl.innerText = "Sistem Aktif - Hazır";
                setCoreState("ONLINE");
                log("Bulut bağlantısı sağlandı.", "sys");
            };

            ws.onclose = () => {
                statusEl.innerText = "Bağlantı koptu, yeniden bağlanılıyor...";
                setCoreState("ERROR");
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                setCoreState("SPEAKING");
                log("JARVIS: " + event.data, "pri");
                setTimeout(() => setCoreState("ONLINE"), 1500);
            };
        }

        function toggleListening() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            isListening = !isListening;
            if (isListening) {
                setCoreState("LISTENING");
                ws.send(JSON.stringify({type: "start_session"}));
            } else {
                setCoreState("ONLINE");
                ws.send(JSON.stringify({type: "stop_session"}));
            }
        }

        function sendCmdText() {
            const val = cmdInput.value.trim();
            if(!val) return;
            log("Siz: " + val, "you");
            cmdInput.value = "";
            setCoreState("THINKING");
            // İstersen metin komutlarını buraya ekleyebilirsin
            setTimeout(() => setCoreState("ONLINE"), 1000);
        }

        function checkEnter(e) {
            if(e.key === 'Enter') sendCmdText();
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
                        turns={"parts": [{"text": "Sistem aktif, kullanıcı seni tetikledi."}]},
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