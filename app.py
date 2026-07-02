from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the saved model package from Google Colab
package = joblib.load('house_model_package.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from frontend
        data = request.json
        input_df = pd.DataFrame([data])
        
        # Inject dummy Id if missing to match original training layout
        if 'Id' not in input_df.columns:
            input_df['Id'] = 0
            
        num_cols = package['num_cols']
        cat_cols = package['cat_cols']
        
        # Explicitly order columns exactly like your training dataset
        input_df = input_df[num_cols + cat_cols]
        
        # 2. Imputation (Failsafe handling for version mismatches)
        try:
            input_df[num_cols] = package['num_imputer'].transform(input_df[num_cols])
            input_df[cat_cols] = package['cat_imputer'].transform(input_df[cat_cols])
        except AttributeError:
            # Modern pandas compatible fallback strategy
            input_df[num_cols] = input_df[num_cols].fillna(0)
            input_df[cat_cols] = input_df[cat_cols].bfill().fillna('Unknown')
        
        # 3. Scale numeric values
        input_df[num_cols] = package['scaler'].transform(input_df[num_cols])
        
        # 4. One-hot encode categorical features
        encoded_cat = package['encoder'].transform(input_df[cat_cols])
        encoded_cat_df = pd.DataFrame(
            encoded_cat, 
            columns=package['encoder'].get_feature_names_out(cat_cols),
            index=input_df.index
        )
        
        # Combine processed columns together
        final_df = input_df.drop(cat_cols, axis=1)
        final_df = pd.concat([final_df, encoded_cat_df], axis=1)
        
        # 5. Predict using the trained model
        prediction = package['model'].predict(final_df)[0]
        
        return jsonify({'price': round(float(prediction), 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)