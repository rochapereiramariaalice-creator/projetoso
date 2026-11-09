import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd

ARQUIVO = "dados/votacao_candidato_munzona_2026_SP.csv"
MUNICIPIO = "BARUERI"
UF = "SP"

N_WORKERS = os.cpu_count() or 2
REPETICOES = 3

COLUNA_VOTOS = "QT_VOTOS_NOMINAIS_VALIDOS"
COLUNAS = ["NM_MUNICIPIO", "SG_UF", "SG_PARTIDO", "DS_CARGO", COLUNA_VOTOS]


def contar_linhas(arquivo):
    with open(arquivo, encoding="latin-1") as f:
        return sum(1 for linha in f) - 1


def somar_por_grupo(dados, coluna):
    return dados.groupby(coluna)[COLUNA_VOTOS].agg(["sum", "count"])


def processar_parte(argumentos):
    arquivo, pular, quantidade = argumentos

    dados = pd.read_csv(
        arquivo,
        sep=";",
        encoding="latin-1",
        usecols=COLUNAS,
        skiprows=range(1, pular + 1),  
        nrows=quantidade,
    )

    dados[COLUNA_VOTOS] = pd.to_numeric(dados[COLUNA_VOTOS], errors="coerce")
    dados = dados.dropna(subset=[COLUNA_VOTOS])

    municipio = dados[(dados["NM_MUNICIPIO"] == MUNICIPIO) & (dados["SG_UF"] == UF)]

    return {
        "soma_geral": dados[COLUNA_VOTOS].sum(),
        "qtd_geral": dados[COLUNA_VOTOS].count(),
        "soma_municipio": municipio[COLUNA_VOTOS].sum(),
        "qtd_municipio": municipio[COLUNA_VOTOS].count(),
        "partidos_geral": somar_por_grupo(dados, "SG_PARTIDO"),
        "cargos_geral": somar_por_grupo(dados, "DS_CARGO"),
        "partidos_municipio": somar_por_grupo(municipio, "SG_PARTIDO"),
        "cargos_municipio": somar_por_grupo(municipio, "DS_CARGO"),
    }


def juntar_tabelas(lista_de_tabelas):
    return pd.concat(lista_de_tabelas).groupby(level=0).sum()


def calcular_medias(tabela):
    medias = tabela["sum"] / tabela["count"]
    return medias.sort_values(ascending=False)


def media(soma, quantidade):
    if quantidade == 0:
        return 0
    return soma / quantidade


def executar_analise(arquivo, tipo_executor):
    inicio_tempo = time.perf_counter()

    total_linhas = contar_linhas(arquivo)
    tamanho_parte = total_linhas // N_WORKERS

    argumentos = []
    for i in range(N_WORKERS):
        pular = i * tamanho_parte
        if i == N_WORKERS - 1:
            quantidade = total_linhas - pular 
        else:
            quantidade = tamanho_parte
        argumentos.append((arquivo, pular, quantidade))

    with tipo_executor(max_workers=N_WORKERS) as executor:
        resultados = list(executor.map(processar_parte, argumentos))

    soma_geral = sum(r["soma_geral"] for r in resultados)
    qtd_geral = sum(r["qtd_geral"] for r in resultados)
    soma_municipio = sum(r["soma_municipio"] for r in resultados)
    qtd_municipio = sum(r["qtd_municipio"] for r in resultados)

    partidos_geral = juntar_tabelas([r["partidos_geral"] for r in resultados])
    cargos_geral = juntar_tabelas([r["cargos_geral"] for r in resultados])
    partidos_municipio = juntar_tabelas([r["partidos_municipio"] for r in resultados])
    cargos_municipio = juntar_tabelas([r["cargos_municipio"] for r in resultados])

    return {
        "media_geral": media(soma_geral, qtd_geral),
        "media_municipio": media(soma_municipio, qtd_municipio),
        "qtd_geral": qtd_geral,
        "qtd_municipio": qtd_municipio,
        "media_partidos_geral": calcular_medias(partidos_geral),
        "media_cargos_geral": calcular_medias(cargos_geral),
        "media_partidos_municipio": calcular_medias(partidos_municipio),
        "media_cargos_municipio": calcular_medias(cargos_municipio),
        "tempo": time.perf_counter() - inicio_tempo,
    }


def rodar_repetido(arquivo, tipo_executor):
    tempos = []
    resultado = None
    for _ in range(REPETICOES):
        resultado = executar_analise(arquivo, tipo_executor)
        tempos.append(resultado["tempo"])
    return resultado, tempos


def imprimir_vencedor(titulo, medias):
    if medias.empty:
        print(titulo + ": sem dados")
    else:
        print(f"{titulo}: {medias.index[0]} (média {medias.iloc[0]:.2f})")


def imprimir_resultados(nome, resultado, tempos):
    print("\n===== RESULTADOS COM " + nome + " =====")

    print(
        f"Média geral do arquivo: {resultado['media_geral']:.2f} "
        f"({resultado['qtd_geral']} registros)"
    )
    print(
        f"Média geral do município ({MUNICIPIO}/{UF}): "
        f"{resultado['media_municipio']:.2f} "
        f"({resultado['qtd_municipio']} registros)"
    )

    print("\n-- Arquivo todo --")
    imprimir_vencedor("Partido com maior média", resultado["media_partidos_geral"])
    imprimir_vencedor("Cargo com maior média", resultado["media_cargos_geral"])

    print(f"\n-- {MUNICIPIO}/{UF} --")
    imprimir_vencedor("Partido com maior média", resultado["media_partidos_municipio"])
    imprimir_vencedor("Cargo com maior média", resultado["media_cargos_municipio"])

    print(f"\nTempos de execução ({len(tempos)} execuções):")
    for i, t in enumerate(tempos, start=1):
        print(f"Execução {i}: {t:.2f} segundos")

    print(f"Tempo total: {sum(tempos):.2f} segundos")
    print(f"Tempo médio por execução: {sum(tempos) / len(tempos):.2f} segundos")


def main():
    arquivo = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO

    if not os.path.exists(arquivo):
        print("Arquivo não encontrado:", arquivo)
        return

    print("Arquivo:", arquivo)
    print("Quantidade de processadores:", N_WORKERS)

    print("\nAquecendo os executores...")
    executar_analise(arquivo, ProcessPoolExecutor)
    executar_analise(arquivo, ThreadPoolExecutor)

    print("\nIniciando análise com Multiprocessing...")
    res_processos, tempos_processos = rodar_repetido(arquivo, ProcessPoolExecutor)
    imprimir_resultados("MULTIPROCESSAMENTO", res_processos, tempos_processos)

    print("\nIniciando análise com Multithreading...")
    res_threads, tempos_threads = rodar_repetido(arquivo, ThreadPoolExecutor)
    imprimir_resultados("MULTITHREADING", res_threads, tempos_threads)

    mesma_media = abs(res_processos["media_geral"] - res_threads["media_geral"]) < 1e-6
    mesmos_partidos = res_processos["media_partidos_geral"].equals(
        res_threads["media_partidos_geral"]
    )
    if mesma_media and mesmos_partidos:
        print("\nVerificação dos resultados: OK!")
    else:
        print("\nATENÇÃO: os resultados dos dois métodos são diferentes!")

    total_processos = sum(tempos_processos)
    total_threads = sum(tempos_threads)

    print("\n===== COMPARAÇÃO FINAL =====")
    print(
        f"Multiprocessing: total {total_processos:.2f} s | "
        f"média {total_processos / REPETICOES:.2f} s"
    )
    print(
        f"Multithreading:  total {total_threads:.2f} s | "
        f"média {total_threads / REPETICOES:.2f} s"
    )

    if total_processos < total_threads:
        print("\nMenor tempo (melhor desempenho): MULTIPROCESSING")
    elif total_threads < total_processos:
        print("\nMenor tempo (melhor desempenho): MULTITHREADING")
    else:
        print("\nOs dois tiveram o mesmo desempenho.")


if __name__ == "__main__":
    main()