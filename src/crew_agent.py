import os
import requests
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

llama_model = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

class DemandForecastInput(BaseModel):
    payload: dict = Field(..., description="ข้อมูลฟีเจอร์ทั้งหมดให้ส่งมาเป็น JSON Object (Dictionary) ตรงๆ ห้ามส่งเป็น String")

class DemandForecastTool(BaseTool):
    name: str = "Demand Forecast Predictor"
    description: str = "เครื่องมือสำหรับพยากรณ์ความต้องการสินค้า รับ Input เป็น Dictionary ของข้อมูลสินค้า"
    args_schema: type[BaseModel] = DemandForecastInput

    def _run(self, payload: dict) -> str:
        url = "http://localhost:8080/predict"
        headers = {"Content-Type": "application/json"}
        
        try:
            # 🌟 โยน payload ที่เป็น dict เข้า API ได้เลย ไม่ต้องแปลง string กลับไปกลับมา
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return f"ผลพยากรณ์ยอดขายสำหรับ {result['Store ID']} สินค้า {result['Product ID']} คือ {result['Predicted_Demand']} ชิ้น"
            else:
                return f"Error: API ตอบกลับมารหัส {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: ไม่สามารถเรียก API ได้ สาเหตุ: {str(e)}"

predict_tool = DemandForecastTool()

inventory_manager = Agent(
    role='Senior Inventory Manager',
    goal='วิเคราะห์และวางแผนระดับสินค้าคงคลังให้เหมาะสมที่สุด ป้องกันของขาดหรือล้นสต็อก',
    backstory='คุณคือผู้เชี่ยวชาญระดับสูงด้าน Supply Chain คุณยึดมั่นในข้อมูลและผลลัพธ์จากการพยากรณ์ของ AI ในการตัดสินใจเสมอ',
    verbose=True,
    allow_delegation=False,
    tools=[predict_tool],
    llm=llama_model
)

mock_situation = """
{
  "Store ID": "Store_1",
  "Product ID": "Prod_10",
  "Category": "Groceries",
  "Region": "North",
  "Inventory Level": 500,
  "Price": 150.0,
  "Discount": 10,
  "Weather Condition": "Sunny",
  "Promotion": 1,
  "Competitor Pricing": 155.0,
  "Seasonality": "High",
  "Epidemic": 0,
  "Day": 21,
  "Month": 5,
  "DayOfWeek": 3,
  "IsWeekend": 0,
  "Quarter": 2,
  "Demand_Lag_1": 125.0,
  "Demand_Lag_7": 110.0,
  "Demand_Rolling_Mean_7": 118.5
}
"""

task_analyze_demand = Task(
    description=f'''
    1. นำข้อมูลต่อไปนี้ส่งเข้าเครื่องมือ 'Demand Forecast Predictor':
    {mock_situation}
    2. เมื่อได้ตัวเลขพยากรณ์มาแล้ว ให้นำมาวิเคราะห์เปรียบเทียบกับ 'Inventory Level' (ระดับคงคลังปัจจุบันที่มีอยู่ 500 ชิ้น)
    3. เขียนสรุปรายงานสั้นๆ เป็นภาษาไทย แนะนำว่ายอดขายจะเป็นอย่างไร และต้องสั่งสินค้าหมวด Groceries ชิ้นนี้เพิ่มหรือไม่
    (สำคัญ: ให้แกะข้อมูลด้านบนแล้วส่งเป็น Dictionary เข้าพารามิเตอร์ชื่อ payload)
    ''',
    expected_output='รายงานสรุปผลพยากรณ์ยอดขาย 1 ย่อหน้า พร้อมคำแนะนำการจัดการคลังสินค้าแบบฟันธง',
    agent=inventory_manager
)

# ==========================================
# 5. ประกอบร่าง Crew และสั่งรัน
# ==========================================
retail_crew = Crew(
    agents=[inventory_manager],
    tasks=[task_analyze_demand],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("🚀 เริ่มต้นกระบวนการ Agentic Workflow ด้วย Llama 3.1 (Local)...")
    result = retail_crew.kickoff()

    print("\n==============================================")
    print("📝 รายงานฉบับสมบูรณ์จาก AI Agent:")
    print("==============================================")
    print(result)