from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uvicorn
import joblib

app = FastAPI()

artifacts = joblib.load("model.pkl")
columns = artifacts["columns"]

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data:dict):
    try:
        df = pd.DataFrame([data])
        # Some CSV headers contain stray whitespace (e.g. a trailing tab), so match on stripped names
        df = df.rename(columns=str.strip)
        df = pd.get_dummies(df, columns=["Course"])
        df = df.reindex(columns=[c.strip() for c in columns], fill_value=0)
        df.columns = columns
        X = artifacts["scaler"].transform(df)
        pred = artifacts["model"].predict(X)
        proba = artifacts["model"].predict_proba(X)[0]
        classes = artifacts["label_encoder"].classes_
        return {
            "prediction": artifacts["label_encoder"].inverse_transform(pred)[0],
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
