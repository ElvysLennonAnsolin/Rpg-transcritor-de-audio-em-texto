import re

import whisper
from pyannote.audio import Pipeline

def transcrever_audio(caminho_audio: str):
    model = whisper.load_model("base")
    resultado = model.transcribe(caminho_audio)

    segmentos = []

    for segmento in resultado ["segments"]:
        segmentos.append({
            "inicio": segmento["start"],
            "fim": segmento["end"],
            "texto": segmento["text"]
        })
        
    print(">>> INICIANDO TRANSCRIÇÃO")
    # código do whisper
    print(">>> TRANSCRIÇÃO FINALIZADA")

    return segmentos, resultado["duration"]


def identificar_vozes(caminho_audio: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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",
    )
    diarizacao = pipeline(caminho_audio)

    falantes = []

    for trecho, _, falante in diarizacao.itertracks(yield_label=True):
        falantes.append({
            "inicio": trecho.start,
            "fim": trecho.end,
            "speaker": falante,
        })
    print(">>> INICIANDO DIARIZAÇÃO")
    # código do pyannote
    print(">>> DIARIZAÇÃO FINALIZADA")
    return falantes


def juntar_falas_com_vozes(segmentos, falantes):
    resultado_final = []

    for segmento in segmentos:
        voz_encontrada = _determinar_falante(segmento, falantes)
        resultado_final.append({
            "voz": voz_encontrada or "DESCONHECIDO",
            "texto": segmento["texto"],
        })

    return resultado_final


def mapear_vozes_para_nomes(segmentos, falantes, duracao_audio, janela_final=20):
    mapa = {}
    inicio_final = max(duracao_audio - janela_final, 0)

    for segmento in segmentos:
        if segmento["inicio"] < inicio_final:
            continue

        nome = _extrair_nome_de_apresentacao(segmento["texto"])
        if not nome:
            continue

        voz = _determinar_falante(segmento, falantes)
        if voz:
            mapa[voz] = nome

    return mapa


def _extrair_nome_de_apresentacao(texto):
    padrao = re.compile(r"eu sou\s+([a-zà-úA-ZÀ-Ú\s]+)", re.IGNORECASE)
    correspondencia = padrao.search(texto)
    if correspondencia:
        return correspondencia.group(1).strip().title()
    return None


def _determinar_falante(segmento, falantes):
    melhor_label = None
    maior_intersecao = 0.0

    for falante in falantes:
        inicio = max(segmento["inicio"], falante["inicio"])
        fim = min(segmento["fim"], falante["fim"])
        intersecao = max(0.0, fim - inicio)

        if intersecao > maior_intersecao:
            maior_intersecao = intersecao
            melhor_label = falante["speaker"]

    return melhor_label