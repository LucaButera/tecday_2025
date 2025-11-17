# TecDay 2025: Data, Code & Clouds

Questa è la repository con il codice per il workshop "Data, Code & Clouds" tenuto durante TecDay 2025.

## Contenuti
Il workshop copre i seguenti argomenti:
- Introduzione al time series forecasting
- Introduzione al dataset PeakWeather
- Confronto tra modelli di forecasting tradizionali e modelli basati su deep learning
- Confronto tra modelli fondazionali e modelli addestrati specificamente sul dataset PeakWeather

## Installazione dell'environment
Per eseguire il codice del workshop, è consigliato creare un ambiente virtuale Python utilizzando `uv`.

Per installare `uv` segui la [documentazione ufficiale](https://docs.astral.sh/uv/getting-started/installation/).

Poi installa le dipendenze eseguendo:
```bash
uv sync
```
dalla root della repository.

>In alternativa è possibile utilizzare un ambiente virtuale `conda`.
>Per installare `conda` segui la [documentazione ufficiale](https://www.anaconda.com/docs/getting-started/miniconda/install).
>
>Poi installa le dipendenze eseguendo:
>```bash
>conda env create -f environment.yml
>```
>dalla root della repository.


## Esecuzione del codice
Il codice del workshop è organizzato in notebook Jupyter. Puoi eseguire i notebook utilizzando Jupyter Notebook.

Per avviare il server Jupyter Notebook, esegui:
```bash
uv run jupyter notebook
```
dalla root della repository.

>Se hai usato `conda` allora esegui:
>````bash
>conda activate tecday-2025
>```

Poi apri nel tuo browser "http://localhost:8888" e seleziona il notebook che desideri eseguire.
