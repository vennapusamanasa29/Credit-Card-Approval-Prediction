from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Load saved files ──
model        = joblib.load('models/card_model.joblib')
encoders     = joblib.load('models/encoders.joblib')
feature_cols = joblib.load('models/feature_cols.joblib')

# ── Print mappings on startup ──
print("\n" + "="*50)
print("MODEL LOADED SUCCESSFULLY")
print("="*50)
print(f"Model    : {type(model).__name__}")
print(f"Features : {feature_cols}")
print("\nENCODER MAPPINGS:")
for col, le in encoders.items():
    print(f"\n  {col}:")
    for i, cls in enumerate(le.classes_):
        print(f"    '{cls}' → {i}")
print("="*50 + "\n")


def safe_encode(col_name, value):
    """
    Encode value using saved LabelEncoder.
    Handles string values like 'ByBirth'
    and numeric strings like '0' or '1'.
    """
    le      = encoders[col_name]
    classes = list(le.classes_)
    value   = str(value).strip()

    if value in classes:
        # Direct match - use encoder
        return int(le.transform([value])[0])
    else:
        # Try numeric fallback
        try:
            numeric_val = int(float(value))
            if 0 <= numeric_val < len(classes):
                print(f"  ⚠️ Numeric fallback: '{value}' → {numeric_val} for {col_name}")
                return numeric_val
        except Exception:
            pass
        print(f"  ⚠️ Unknown value '{value}' for {col_name}. Classes: {classes}. Using 0.")
        return 0


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        print("\n" + "="*50)
        print("📥 NEW PREDICTION REQUEST")
        print("="*50)
        print("Raw input:", data)

        # ── Encode all inputs ──
        gender_enc   = safe_encode('Gender',   data['gender'])
        industry_enc = safe_encode('Industry', data['industry'])
        citizen_enc  = safe_encode('Citizen',  data['citizen'])

        print(f"\nEncoded:")
        print(f"  Gender   : '{data['gender']}' → {gender_enc}")
        print(f"  Industry : '{data['industry']}' → {industry_enc}")
        print(f"  Citizen  : '{data['citizen']}' → {citizen_enc}")

        # ── Build input in exact feature order ──
        input_data = {
            'Gender':        gender_enc,
            'Age':           float(data['age']),
            'Debt':          float(data['debt']),
            'Married':       int(data['married']),
            'BankCustomer':  int(data['bank_customer']),
            'Industry':      industry_enc,
            'YearsEmployed': float(data['years_employed']),
            'PriorDefault':  int(data['prior_default']),
            'Employed':      int(data['employed']),
            'CreditScore':   float(data['credit_score']),
            'Citizen':       citizen_enc,
            'Income':        float(data['income'])
        }

        print(f"\nFull input dict: {input_data}")

        # ── Create DataFrame in correct column order ──
        input_df    = pd.DataFrame([input_data])[feature_cols]
        print(f"\nFeature array:\n{input_df.to_string()}")

        # ── Predict ──
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence  = round(float(max(probability)) * 100, 1)

        approved = bool(prediction == 1)

        print(f"\n🎯 Result     : {'APPROVED' if approved else 'REJECTED'}")
        print(f"📊 Confidence : {confidence}%")
        print(f"📊 Proba      : Rejected={probability[0]:.3f} | Approved={probability[1]:.3f}")
        print("="*50 + "\n")

        return jsonify({
            'approved':   approved,
            'confidence': confidence
        })

    except Exception as e:
        import traceback
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)