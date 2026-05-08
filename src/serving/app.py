import os
import pickle
import pandas as pd
import litserve as ls
from typing import List, Optional
from pydantic import BaseModel


class Passenger(BaseModel):
    Pclass: str
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Fare: Optional[float] = None
    Embarked: Optional[str] = None

class PredictRequest(BaseModel):
    passengers: List[Passenger]


class TitanicAPI(ls.LitAPI):

    def setup(self, device):
        MODEL_PATH = "models/xgb-cat"
        PROCESSED_PATH = "data/processed"

        with open(os.path.join(MODEL_PATH, "final_xgb_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(PROCESSED_PATH, "preprocessing_pipeline.pkl"), "rb") as f:
            self.pipeline = pickle.load(f)

        with open(os.path.join(MODEL_PATH, "model_target_translator.pkl"), "rb") as f:
            self.translator = pickle.load(f)

    def decode_request(self, request: PredictRequest):
        df = pd.DataFrame([p.model_dump() for p in request.passengers])
        return df

    def predict(self, df: pd.DataFrame):
        df_processed = self.pipeline.transform(df)
        preds_enc = self.model.predict(df_processed)
        return preds_enc

    def encode_response(self, preds):
        decoder = self.translator["decoder"]
        predictions = [int(decoder[int(p)]) for p in preds]
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }


if __name__ == "__main__":
    api = TitanicAPI(api_path="/predict")
    server = ls.LitServer(api)
    server.run(port=8000)