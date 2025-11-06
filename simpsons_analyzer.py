import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import json

def load_simpsons_data():
    """Load all your Simpsons CSV files"""
    base_path = r"C:\Users\Browndan\Downloads\simpsons\simpsons"
    
    files = {
        'characters': r"simpsons_characters.csv",
        'episodes': r"simpsons_episodes.csv", 
        'locations': r"simpsons_locations.csv",
        'script_lines': r"simpsons_script_lines.csv"
    }
    
    data = {}
    for key, filename in files.items():
        full_path = os.path.join(base_path, filename)
        try:
            data[key] = pd.read_csv(full_path, low_memory=False)
            print(f"✅ Loaded {key}: {data[key].shape}")
            print(f"   Columns: {list(data[key].columns)}")
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")
    
    return data

def create_automated_features(data):
    """Automatically discover column names and create features"""
    
    episodes = data['episodes'].copy()
    print(f"\n📊 Episodes data columns: {list(episodes.columns)}")
    print("First few rows of episodes data:")
    print(episodes.head(3))
    
    episode_id_col = 'id'
    print(f"🔗 Using episode ID column: {episode_id_col}")
    
    # Find potential target columns
    potential_targets = []
    for col in episodes.columns:
        if any(term in col.lower() for term in ['rating', 'viewer', 'score', 'imdb', 'views']):
            potential_targets.append(col)
    
    print(f"🎯 Potential target columns: {potential_targets}")
    
    # Basic feature engineering
    feature_columns = []
    
    # Add basic numeric columns (excluding potential targets and IDs)
    numeric_cols = episodes.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = potential_targets + [episode_id_col, 'number_in_series']
    feature_columns.extend([col for col in numeric_cols if col not in exclude_cols])
    
    # Create time-based features from original_air_date
    if 'original_air_date' in episodes.columns:
        try:
            episodes['airdate'] = pd.to_datetime(episodes['original_air_date'], errors='coerce')
            episodes['year'] = episodes['airdate'].dt.year
            episodes['month'] = episodes['airdate'].dt.month
            episodes['day_of_week'] = episodes['airdate'].dt.dayofweek
            feature_columns.extend(['year', 'month', 'day_of_week'])
            print(f"📅 Created time features from original_air_date")
        except Exception as e:
            print(f"⚠️ Could not parse date column: {e}")
    
    # Merge with script data
    if 'script_lines' in data:
        script = data['script_lines']
        print(f"\n📜 Script data columns: {list(script.columns)}")
        
        if 'episode_id' in script.columns:
            # Count unique characters per episode
            if 'character_id' in script.columns:
                char_counts = script.groupby('episode_id')['character_id'].nunique().reset_index()
                char_counts.columns = ['episode_id', 'unique_characters']
                episodes = episodes.merge(char_counts, left_on='id', right_on='episode_id', how='left')
                episodes.drop('episode_id', axis=1, inplace=True)
                feature_columns.append('unique_characters')
                print("👥 Added character count feature")
            
            # Count total lines per episode
            line_counts = script.groupby('episode_id').size().reset_index(name='total_lines')
            episodes = episodes.merge(line_counts, left_on='id', right_on='episode_id', how='left')
            episodes.drop('episode_id', axis=1, inplace=True)
            feature_columns.append('total_lines')
            print("💬 Added line count feature")
            
            # Count speaking lines per episode
            if 'speaking_line' in script.columns:
                speaking_counts = script[script['speaking_line'] == True].groupby('episode_id').size().reset_index(name='speaking_lines')
                episodes = episodes.merge(speaking_counts, left_on='id', right_on='episode_id', how='left')
                episodes.drop('episode_id', axis=1, inplace=True)
                feature_columns.append('speaking_lines')
                print("🗣️ Added speaking lines feature")
            
            # Average word count per line
            if 'word_count' in script.columns:
                avg_words = script.groupby('episode_id')['word_count'].mean().reset_index(name='avg_words_per_line')
                episodes = episodes.merge(avg_words, left_on='id', right_on='episode_id', how='left')
                episodes.drop('episode_id', axis=1, inplace=True)
                feature_columns.append('avg_words_per_line')
                print("📝 Added average word count feature")
    
    # Add season and episode number if they exist
    for col in ['season', 'number_in_season']:
        if col in episodes.columns and col not in feature_columns:
            feature_columns.append(col)
    
    print(f"\n🎲 Final feature columns: {feature_columns}")
    return episodes, feature_columns, potential_targets

def train_automated_model(episodes, feature_columns, target_column='imdb_rating'):
    """Train a model that will automatically find important factors"""
    
    # Handle missing values
    model_data = episodes[feature_columns + [target_column]].copy()
    
    print(f"\n📊 Data before cleaning: {len(model_data)} rows")
    print(f"Missing values per column:")
    print(model_data.isnull().sum())
    
    # Fill missing values instead of dropping all rows
    for col in feature_columns:
        if model_data[col].isnull().any():
            model_data.loc[:, col] = model_data[col].fillna(model_data[col].median())
    
    # Drop rows where target is missing
    model_data = model_data.dropna(subset=[target_column])
    
    print(f"📊 Data after cleaning: {len(model_data)} rows")
    
    if len(model_data) < 20:
        print("❌ Not enough data available after removing missing values")
        return None, None, None, None, None, None
    
    X = model_data[feature_columns]
    y = model_data[target_column]
    
    # Split chronologically (last 20% as test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"📚 Training on {len(X_train)} episodes, testing on {len(X_test)}")
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training R²: {train_score:.3f}")
    print(f"   Test R²: {test_score:.3f}") 
    print(f"   Test MAE: {mae:.3f}")
    
    return model, X.columns, X_test, y_test, X_train, y_train

def generate_html_dashboard(results):
    """Generate HTML dashboard with embedded results"""
    results_json = json.dumps(results, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simpsons Analysis Results</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 30px; }}
        .header h1 {{ font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); transition: transform 0.3s ease; }}
        .card:hover {{ transform: translateY(-5px); }}
        .card h2 {{ color: #667eea; margin-bottom: 20px; font-size: 1.5em; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; align-items: center; padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; }}
        .metric-label {{ font-weight: 600; color: #333; }}
        .metric-value {{ font-size: 1.3em; font-weight: bold; color: #667eea; }}
        .feature-bar {{ margin: 10px 0; }}
        .feature-name {{ font-weight: 500; margin-bottom: 5px; color: #333; }}
        .bar-container {{ background: #e0e7ff; height: 30px; border-radius: 15px; overflow: hidden; position: relative; }}
        .bar-fill {{ height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold; font-size: 0.9em; transition: width 1s ease; }}
        .prediction-row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; padding: 12px; margin: 8px 0; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
        .prediction-row.good {{ border-left-color: #10b981; }}
        .prediction-row.bad {{ border-left-color: #ef4444; }}
        .prediction-item {{ text-align: center; }}
        .prediction-label {{ font-size: 0.85em; color: #666; margin-bottom: 3px; }}
        .prediction-value {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .status-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; font-weight: 600; margin: 5px; background: #d1fae5; color: #065f46; }}
        .chart-container {{ position: relative; height: 300px; margin-top: 20px; }}
        .full-width {{ grid-column: 1 / -1; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .card {{ animation: fadeIn 0.5s ease forwards; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Simpsons Episode Analysis</h1>
            <p>Machine Learning Insights Dashboard</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Model Performance</h2>
                <div class="metric">
                    <span class="metric-label">Training R²</span>
                    <span class="metric-value" id="train-r2">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Test R²</span>
                    <span class="metric-value" id="test-r2">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mean Absolute Error</span>
                    <span class="metric-value" id="mae">-</span>
                </div>
                <div style="margin-top: 15px;">
                    <span class="status-badge">Model Ready</span>
                </div>
            </div>
            
            <div class="card">
                <h2>📁 Dataset Information</h2>
                <div class="metric">
                    <span class="metric-label">Total Episodes</span>
                    <span class="metric-value" id="total-episodes">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Training Episodes</span>
                    <span class="metric-value" id="train-episodes">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Test Episodes</span>
                    <span class="metric-value" id="test-episodes">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Features Used</span>
                    <span class="metric-value" id="feature-count">-</span>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>🎯 Most Important Features</h2>
                <p style="color: #666; margin-bottom: 15px;">What factors most influence episode ratings?</p>
                <div id="feature-list"></div>
            </div>
            
            <div class="card full-width">
                <h2>📈 Prediction Accuracy</h2>
                <div class="chart-container">
                    <canvas id="predictionChart"></canvas>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>🎲 Sample Predictions</h2>
                <div id="predictions-list"></div>
            </div>
        </div>
    </div>
    
    <script>
        const analysisResults = {results_json};
        
        document.getElementById('train-r2').textContent = analysisResults.modelPerformance.trainR2.toFixed(3);
        document.getElementById('test-r2').textContent = analysisResults.modelPerformance.testR2.toFixed(3);
        document.getElementById('mae').textContent = analysisResults.modelPerformance.mae.toFixed(3);
        
        document.getElementById('total-episodes').textContent = analysisResults.datasetInfo.totalEpisodes;
        document.getElementById('train-episodes').textContent = analysisResults.datasetInfo.trainEpisodes;
        document.getElementById('test-episodes').textContent = analysisResults.datasetInfo.testEpisodes;
        document.getElementById('feature-count').textContent = analysisResults.datasetInfo.featureCount;
        
        const featureList = document.getElementById('feature-list');
        analysisResults.featureImportance.forEach(feature => {{
            const percentage = (feature.importance * 100).toFixed(1);
            const featureDiv = document.createElement('div');
            featureDiv.className = 'feature-bar';
            featureDiv.innerHTML = `
                <div class="feature-name">${{feature.name}}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${{percentage}}%">${{percentage}}%</div>
                </div>
            `;
            featureList.appendChild(featureDiv);
        }});
        
        const predictionsList = document.getElementById('predictions-list');
        analysisResults.predictions.forEach((pred, idx) => {{
            const errorClass = pred.error < 0.3 ? 'good' : 'bad';
            const predDiv = document.createElement('div');
            predDiv.className = `prediction-row ${{errorClass}}`;
            predDiv.innerHTML = `
                <div class="prediction-item">
                    <div class="prediction-label">Predicted</div>
                    <div class="prediction-value">${{pred.predicted.toFixed(2)}}</div>
                </div>
                <div class="prediction-item">
                    <div class="prediction-label">Actual</div>
                    <div class="prediction-value">${{pred.actual.toFixed(2)}}</div>
                </div>
                <div class="prediction-item">
                    <div class="prediction-label">Error</div>
                    <div class="prediction-value">${{pred.error.toFixed(2)}}</div>
                </div>
            `;
            predictionsList.appendChild(predDiv);
        }});
        
        const ctx = document.getElementById('predictionChart').getContext('2d');
        const minVal = Math.min(...analysisResults.predictions.map(p => Math.min(p.actual, p.predicted))) - 0.5;
        const maxVal = Math.max(...analysisResults.predictions.map(p => Math.max(p.actual, p.predicted))) + 0.5;
        
        new Chart(ctx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Predictions vs Actual',
                    data: analysisResults.predictions.map(p => ({{
                        x: p.actual,
                        y: p.predicted
                    }})),
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 12
                }}, {{
                    label: 'Perfect Prediction',
                    data: [{{x: minVal, y: minVal}}, {{x: maxVal, y: maxVal}}],
                    type: 'line',
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Actual Rating',
                            font: {{ size: 14, weight: 'bold' }}
                        }},
                        min: minVal,
                        max: maxVal
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Predicted Rating',
                            font: {{ size: 14, weight: 'bold' }}
                        }},
                        min: minVal,
                        max: maxVal
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return `Predicted: ${{context.parsed.y.toFixed(2)}}, Actual: ${{context.parsed.x.toFixed(2)}}`;
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
    return html

def complete_automated_analysis():
    """One function to run everything automatically"""
    
    print("🚀 Starting Automated Simpsons Analysis...")
    
    # Load data
    data = load_simpsons_data()
    
    if 'episodes' not in data:
        print("❌ Could not load episodes data - check file paths")
        return None, None
    
    # Create features automatically
    episodes, feature_columns, potential_targets = create_automated_features(data)
    
    if not feature_columns:
        print("❌ No features found to train on")
        return None, None
    
    # Choose target (prefer IMDB rating)
    target_col = None
    for preferred in ['imdb_rating', 'us_viewers_in_millions', 'views']:
        if preferred in episodes.columns:
            target_col = preferred
            break
    
    if not target_col:
        print("❌ No suitable target column found")
        return None, None
    
    print(f"🎯 Using target column: {target_col}")
    
    # Train model - this returns 6 values
    model, features, X_test, y_test, X_train, y_train = train_automated_model(episodes, feature_columns, target_col)
    
    if model is None:
        print("\n❌ Model training failed")
        return None, None
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Model-Discovered Important Factors:")
    print("=" * 50)
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']:.<35} {row['importance']:.4f}")
    
    # Sample predictions
    predictions_list = []
    if X_test is not None and len(X_test) > 0:
        print(f"\n🎯 Sample Predictions:")
        for i in range(min(5, len(X_test))):
            sample_pred = model.predict(X_test.iloc[[i]])[0]
            actual = y_test.iloc[i]
            error = abs(sample_pred - actual)
            predictions_list.append({
                'predicted': float(sample_pred),
                'actual': float(actual),
                'error': float(error)
            })
            print(f"   Episode {i+1}: Predicted={sample_pred:.2f}, Actual={actual:.2f}, Error={error:.2f}")
    
    # Calculate metrics for JSON
    train_r2 = float(model.score(X_train, y_train))
    test_r2 = float(model.score(X_test, y_test))
    test_mae = float(mean_absolute_error(y_test, model.predict(X_test)))
    
    # Save results to JSON for dashboard
    results = {
        'modelPerformance': {
            'trainR2': train_r2,
            'testR2': test_r2,
            'mae': test_mae
        },
        'datasetInfo': {
            'totalEpisodes': len(X_train) + len(X_test),
            'trainEpisodes': len(X_train),
            'testEpisodes': len(X_test),
            'featureCount': len(features)
        },
        'featureImportance': [
            {'name': row['feature'], 'importance': float(row['importance'])}
            for _, row in feature_importance.iterrows()
        ],
        'predictions': predictions_list
    }
    
    # Generate complete HTML dashboard with embedded results
    html_content = generate_html_dashboard(results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simpsons_analysis.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n💾 Dashboard saved to: {output_path}")
    print(f"   Open this file in your browser to view results!")
    
    # Also save JSON for reference
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, episodes

# RUN THE COMPLETE ANALYSIS
if __name__ == "__main__":
    model, data = complete_automated_analysis()
    
    if model is not None:
        print("\n✅ Analysis complete! The model has automatically discovered")
        print("   which factors influence Simpsons episode popularity.")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")