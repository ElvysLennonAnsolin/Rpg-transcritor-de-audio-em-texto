from fastapi import FastAPI
from fastapi import UploadFile, File
import os
import shutil 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from audio_processor import identificar_vozes
from audio_processor import (
    transcrever_audio,
    juntar_falas_com_vozes,
    mapear_vozes_para_nomes,
)




app=FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"saved_as": file_path}

@app.post("/processar")
async def processar_audio(file: UploadFile = File(...)):
    caminho = f"uploads/{file.filename}"

    with open(caminho, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segmentos, duracao = transcrever_audio(caminho)
    falantes = identificar_vozes(caminho)

    falas = juntar_falas_com_vozes(segmentos, falantes)
    mapa_vozes = mapear_vozes_para_nomes(segmentos, falantes, duracao)

    texto_final = ""
    for fala in falas:
        identificador = mapa_vozes.get(fala["voz"], fala["voz"])
        texto_final += f"{identificador}: {fala['texto']}\n"

    return {"texto": texto_final}