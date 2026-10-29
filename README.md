# Student Academic Risk

Machine-Learning-Projekt, das vorhersagt, ob Studierende ihr Studium **abbrechen** (Dropout), **weiter eingeschrieben** bleiben (Enrolled) oder **abschließen** (Graduate). Das Modell wird über eine FastAPI mit Web-UI bereitgestellt und läuft in Docker.

## Features

- Random-Forest-Modell mit SMOTE (Klassenausgleich) und GridSearch-Tuning
- Experiment-Tracking mit MLflow
- REST-API mit FastAPI (`/predict`, `/health`)
- Web-UI zur Eingabe von Studierendendaten
- Docker-Image und CI-Pipeline mit GitHub Actions

## Daten

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI): 4.424 Studierende, 36 Merkmale, z. B. Herkunft, Zulassungsnote, Finanzen, Studienleistungen im 1. und 2. Semester und Wirtschaftslage.

## Modell

| Klasse    | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Dropout   | 0.86      | 0.73   | 0.79 |
| Enrolled  | 0.48      | 0.48   | 0.48 |
| Graduate  | 0.80      | 0.89   | 0.84 |

**Accuracy: 76 %** auf dem Testset (20 %). Die Klasse *Enrolled* ist am schwersten vorherzusagen, weil diese Studierenden sich noch in keine Richtung entwickelt haben.

## Projektstruktur

```
├── data/data.csv          # Datensatz
├── exploration.ipynb      # EDA, Training, MLflow-Tracking
├── model.pkl              # Modell + Scaler + Spalten + Label-Encoder
├── main.py                # FastAPI-App
├── static/index.html      # Web-UI
├── tests/                 # API-Tests
├── Dockerfile
└── .github/workflows/     # CI-Pipeline
```

## Starten

### Mit Docker

```bash
docker build -t student-risk-api .
docker run -p 8000:8000 student-risk-api
```

### Lokal (mit [uv](https://docs.astral.sh/uv/))

```bash
uv sync
uv run main.py
```

Danach erreichbar unter:

- **UI:** http://localhost:8000
- **API-Dokumentation:** http://localhost:8000/docs

## API

`POST /predict` erwartet alle 36 Merkmale als JSON. Ein vollständiges Beispiel liegt in [`tests/sample_student.json`](tests/sample_student.json).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_student.json
```

```json
{
  "prediction": "Graduate",
  "probabilities": { "Dropout": 0.075, "Enrolled": 0.075, "Graduate": 0.85 }
}
```

## Modell neu trainieren

1. `exploration.ipynb` öffnen und alle Zellen ausführen. Das überschreibt `model.pkl`.
2. MLflow-Ergebnisse ansehen: `uv run mlflow ui`

## Tests

```bash
uv run pytest
```

Die GitHub-Action führt bei jedem Push auf `main` die Tests aus, baut das Docker-Image und prüft den laufenden Container mit einem Testrequest.

## Tech Stack

Python 3.12 · scikit-learn · imbalanced-learn · pandas · MLflow · FastAPI · Docker · GitHub Actions · uv
