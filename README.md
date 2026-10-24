# AmoPaw BioAge 模型训练与部署原型

> **重要限制：** 模型只用于验证数据处理、训练、API 和部署流程，不得将当前指标描述为真实宠物生理年龄准确率，也不应用于兽医诊断或健康决策。

## 目录

- `AmoPaw_BioAge_synthetic_dataset.csv`：输入数据（请从数据包根目录复制到本目录或训练时指定路径）
- `train_model.py`：按宠物聚合记录、训练 Ridge 与 CatBoost、选择验证集 MAE 较低者、测试集评估
- `app.py`：FastAPI 推理服务
- `model_metadata.joblib` / `bioage_catboost.cbm` 或 `bioage_model.joblib`：当前已训练模型及元数据
- `metrics.json`：本次训练的验证/测试指标
- `pet_level_features_synthetic.csv`：聚合到宠物级的特征表
- `requirements.txt`、`Dockerfile`、`docker-compose.yml`：部署文件

## 本次训练摘要

- 宠物总数：240
- 有年龄标签的宠物：211
- 训练/验证/测试有标签宠物：148 / 30 / 33
- 验证集选出的模型：CatBoost
- 测试集指标：MAE=2.4512 年，RMSE=3.0034 年，R²=-0.2145

以上指标仅代表合成数据上的流程测试结果。

## 本地启动

使用 Python 3.11。

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

打开 `http://127.0.0.1:8000/docs` 查看交互式 API 文档；健康检查为 `GET /health`。

## Docker 部署

将 `app.py`、`requirements.txt`、`Dockerfile`、`docker-compose.yml` 和本次生成的模型文件放在同一目录，然后执行：

```bash
docker compose up --build -d
docker compose logs -f
```

服务监听 `8000` 端口。正式部署时应放在 HTTPS 反向代理后面，并配置访问认证、限流、日志脱敏、模型版本管理和监控。

## API 示例

`POST /api/v1/bioage/predict`

```json
{
  "pet_id": "AP_DEMO_001",
  "species": "dog",
  "sex": "母",
  "neutered": "是",
  "breed": "柯基",
  "weight_kg": 12.5,
  "respiratory_rate_median": 20.0,
  "respiratory_rate_std": 2.0,
  "respiratory_valid_rate": 0.9,
  "heart_rate_median": 80.0,
  "heart_rate_std": 4.0,
  "heart_rate_valid_rate": 0.9,
  "hrv_rmssd_median": 45.0,
  "hrv_rmssd_std": 6.0,
  "hrv_valid_rate": 0.8,
  "high_signal_quality_ratio": 0.9,
  "valid_measurement_ratio": 0.9,
  "valid_record_count": 3,
  "chronological_age_years": 5.0
}
```

## 必要工作

1. 使用真实宠物个体建立标签：年龄、物种、品种、体重、性别/绝育状态，以及可追溯的兽医健康信息。
2. 验证雷达派生的心率、呼吸率与参考设备的一致性；HRV 特征需先证明信号质量与参考 ECG 的一致性，未验证时不要输入模型。
3. 以宠物 ID 做分组划分，另设按时间或外部机构的独立验证集。
4. 需要真正的生物年龄标签或可验证的健康结局（例如纵向健康变化）才能训练/验证“生理年龄”；仅预测实际年龄只能称为年龄估计。
5. 用真实训练集重新训练，并进行误差分层分析（犬/猫、年龄段、品种、信号质量）；在验证通过前仅展示为探索性功能。
6. API 的输入是聚合特征，不是原始雷达 ADC/FFT 数据。生产环境需由信号处理模块先生成并校验这些特征。
