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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>J.A.R.V.I.S // WINDOWS VOICE CORE</title>
    <style>
        :root {
            --c-bg: #020c0c;
            --c-pri: #00d4c0;
            --c-mid: #006a62;
            --c-dim: #0a2a28;
            --c-text: #7dfff6;
            --c-green: #00ff88;
            --c-blue: #4488ff;
            --c-gold: #ffcc00;
            --c-red: #ff3344;
        }

        body {
            background-color: var(--c-bg);
            color: var(--c-text);
            font-family: 'Courier New', Courier, monospace;
            margin: 0;
            padding: 10px;
            display: flex;
            flex-direction: column;
            height: 98vh;
            box-sizing: border-box;
            overflow: hidden;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--c-mid);
            padding-bottom: 8px;
            font-size: 13px;
            letter-spacing: 1px;
        }

        .header .badge {
            color: var(--c-pri);
            font-weight: bold;
        }

        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .arc-reactor {
            width: 240px;
            height: 240px;
            border: 2px dashed var(--c-pri);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            animation: spin 18s linear infinite;
            box-shadow: 0 0 25px rgba(0, 212, 192, 0.2);
            margin: 10px 0;
            cursor: pointer;
        }

        .arc-reactor::before {
            content: '';
            position: absolute;
            width: 200px;
            height: 200px;
            border: 1px solid var(--c-mid);
            border-radius: 50%;
            animation: spin-reverse 12s linear infinite;
        }

        .inner-core {
            width: 130px;
            height: 130px;
            border: 2px solid var(--c-pri);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px rgba(0, 212, 192, 0.4);
        }

        .core-dot {
            width: 35px;
            height: 35px;
            background-color: var(--c-pri);
            border-radius: 50%;
            box-shadow: 0 0 20px var(--c-pri);
            transition: background-color 0.3s, box-shadow 0.3s;
        }

        .core-dot.listening { background-color: var(--c-green); box-shadow: 0 0 25px var(--c-green); }
        .core-dot.speaking { background-color: var(--c-blue); box-shadow: 0 0 25px var(--c-blue); }
        .core-dot.thinking { background-color: var(--c-gold); box-shadow: 0 0 25px var(--c-gold); }

        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes spin-reverse { 100% { transform: rotate(-360deg); } }

        .chat-box {
            width: 100%;
            flex: 1;
            background: rgba(3, 15, 15, 0.8);
            border: 1px solid var(--c-mid);
            border-radius: 2px;
            padding: 10px;
            overflow-y: auto;
            font-size: 12px;
            margin-bottom: 10px;
            max-height: 160px;
        }

        .chat-box div {
            margin-bottom: 6px;
            line-height: 1.4;
        }

        .controls {
            display: flex;
            gap: 6px;
            width: 100%;
        }

        input {
            flex: 1;
            background: #020c0c;
            border: 1px solid var(--c-mid);
            color: var(--c-text);
            padding: 10px;
            border-radius: 2px;
            font-family: inherit;
            font-size: 13px;
        }

        input:focus {
            border-color: var(--c-pri);
            outline: none;
        }

        button {
            background: var(--c-mid);
            color: var(--c-text);
            border: 1px solid var(--c-pri);
            padding: 10px 14px;
            font-weight: bold;
            border-radius: 2px;
            cursor: pointer;
            font-family: inherit;
        }

        button:active {
            background: var(--c-pri);
            color: var(--c-bg);
        }

        button.mic-btn {
            background: #1a0808;
            border-color: var(--c-red);
            color: var(--c-red);
            font-size: 16px;
        }

        button.mic-btn.listening {
            background: var(--c-red);
            color: white;
            animation: pulse 1s infinite;
        }

        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    </style>
</head>
<body>

    <div class="header">
        <span>J.A.R.V.I.S // CORE</span>
        <span class="badge" id="status-text">ONLINE</span>
    </div>

    <div class="main-container">
        <div class="arc-reactor" onclick="toggleMic()">
            <div class="inner-core">
                <div class="core-dot" id="core-dot"></div>
            </div>
        </div>

        <div class="chat-box" id="chat-box">
            <div><span style="color:var(--c-gold)">[SYS]</span> J.A.R.V.I.S Bulut Çekirdeği Aktif. Dinlemedeyim...</div>
        </div>
    </div>

    <div class="controls">
        <input type="text" id="cmd-input" placeholder="Sistem komutu girin..." onkeypress="checkEnter(event)">
        <button id="mic-btn" class="mic-btn" onclick="toggleMic()">🎤</button>
        <button onclick="sendCmd()">GÖNDER</button>
    </div>

    <script>
        let ws = null;
        const chatBox = document.getElementById("chat-box");
        const cmdInput = document.getElementById("cmd-input");
        const micBtn = document.getElementById("mic-btn");
        const coreDot = document.getElementById("core-dot");
        const statusText = document.getElementById("status-text");

        function connectWs() {
            const proto = location.protocol === "https:" ? "wss://" : "ws://";
            ws = new WebSocket(proto + location.host + "/ws");
            ws.onmessage = (event) => {
                setCoreState("SPEAKING");
                appendLog("JARVIS", event.data, "pri");
                speak(event.data);
            };
            ws.onclose = () => setTimeout(connectWs, 3000);
        }
        connectWs();

        function appendLog(sender, text, type="pri") {
            const div = document.createElement("div");
            let col = "var(--c-text)";
            if(type === "sys") col = "var(--c-gold)";
            if(type === "you") col = "#d0f0ee";
            if(type === "err") col = "var(--c-red)";
            
            div.innerHTML = `<strong style="color:${col}">[${new Date().toLocaleTimeString()}] ${sender}:</strong> ${text}`;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function setCoreState(state) {
            coreDot.className = "core-dot " + state.toLowerCase();
            statusText.innerText = state;
        }

        function sendCmd() {
            const text = cmdInput.value.trim();
            if(!text) return;
            appendLog("Siz", text, "you");
            cmdInput.value = "";

            setCoreState("THINKING");
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: "start_session", command: text}));
            } else {
                appendLog("ERR", "Websocket bağlı değil!", "err");
                setCoreState("ONLINE");
            }
        }

        function checkEnter(e) {
            if(e.key === 'Enter') sendCmd();
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'tr-TR';
                utterance.onend = () => setCoreState("ONLINE");
                window.speechSynthesis.speak(utterance);
            } else {
                setCoreState("ONLINE");
            }
        }

        let recognition;
        let isListening = false;
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.lang = 'tr-TR';
            recognition.continuous = false;

            recognition.onresult = function(event) {
                const speechToText = event.results[0][0].transcript;
                cmdInput.value = speechToText;
                micBtn.classList.remove("listening");
                isListening = false;
                setCoreState("ONLINE");
                sendCmd();
            };

            recognition.onerror = function() {
                micBtn.classList.remove("listening");
                isListening = false;
                setCoreState("ONLINE");
            };

            recognition.onend = function() {
                micBtn.classList.remove("listening");
                isListening = false;
                if(statusText.innerText === "LISTENING") setCoreState("ONLINE");
            };
        }

        function toggleMic() {
            if (!recognition) {
                alert("Tarayıcınız ses tanımayı desteklemiyor.");
                return;
            }
            if (isListening) {
                recognition.stop();
                micBtn.classList.remove("listening");
                isListening = false;
                setCoreState("ONLINE");
            } else {
                recognition.start();
                micBtn.classList.add("listening");
                isListening = true;
                setCoreState("LISTENING");
            }
        }
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
                
                user_msg = packet.get("command") or "Sistem aktif, kullanıcı seni tetikledi."
                await session.send_client_content(
                    turns={"parts": [{"text": user_msg}]},
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
