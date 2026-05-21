import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Retail Store Demand Forecasting API", version="1.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_lgbm_model.pkl")
model_pipeline = joblib.load(MODEL_PATH)

# feature data in pipeline
class PredictionRequest(BaseModel):
    Store_ID: str = Field(..., alias="Store ID", description="รหัสร้านค้า เช่น Store_1")
    Product_ID: str = Field(..., alias="Product ID", description="รหัสสินค้า เช่น Prod_10")
    Category: str = Field(..., description="หมวดหมู่สินค้า เช่น Groceries")
    Region: str = Field(..., description="ภูมิภาค เช่น North")
    Price: float = Field(..., description="ราคาสินค้า ณ วันนั้น")
    Discount: int = Field(..., description="เปอร์เซ็นต์ส่วนลด")
    Weather_Condition: str = Field(..., alias="Weather Condition", description="สภาพอากาศ เช่น Sunny")
    Promotion: int = Field(..., description="จัดโปรโมชันหรือไม่ (0 หรือ 1)")
    Competitor_Pricing: float = Field(..., alias="Competitor Pricing", description="ราคาสินค้าของคู่แข่ง")
    Seasonality: str = Field(..., description="ฤดูกาล เช่น High")
    Epidemic: int = Field(..., description="ช่วงโรคระบาดหรือไม่ (0 หรือ 1)")
    Inventory_Level: int = Field(..., alias="Inventory Level", description="จำนวนสินค้าคงคลังในปัจจุบัน")
    
    # feature from data engineering
    Day: int = Field(..., description="วันที่ (1-31)")
    Month: int = Field(..., description="เดือน (1-12)")
    DayOfWeek: int = Field(..., description="วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)")
    IsWeekend: int = Field(..., description="เป็นวันหยุดสุดสัปดาห์ไหม (0 หรือ 1)")
    Quarter: int = Field(..., description="ไตรมาสที่ (1-4)")
    
    # feature from lag and rolling
    Demand_Lag_1: float = Field(..., description="ยอดขายของเมื่อวาน")
    Demand_Lag_7: float = Field(..., description="ยอดขายของสัปดาห์ที่แล้ว")
    Demand_Rolling_Mean_7: float = Field(..., description="ยอดขายเฉลี่ย 7 วันย้อนหลัง")

    class Config:
        populate_by_name = True

# endpoint for health check
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Retail Demand Forecasting API พร้อมทำงานแล้วครับ!"}


# endpoint for prediction
@app.post("/predict", summary="พยากรณ์ยอดความต้องการสินค้า (Demand)")
def predict_demand(request: PredictionRequest):
    try:
        # transform request data to dict with alias keys (to match the original feature names in the pipeline)
        data_dict = request.model_dump(by_alias=True)
        
        # transform dict to DataFrame
        input_df = pd.DataFrame([data_dict])
        
        # send the input data through the pipeline to get prediction
        prediction = model_pipeline.predict(input_df)
        
        # extract the predicted value
        predicted_value = float(prediction[0])
        
        # prevent negative demand
        final_demand = max(0.0, predicted_value)
        
        # return the response with predicted demand, rounded to 2 decimal places
        return {
            "Store ID": request.Store_ID,
            "Product ID": request.Product_ID,
            "Predicted_Demand": round(final_demand, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
