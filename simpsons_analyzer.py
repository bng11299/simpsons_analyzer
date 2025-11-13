import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import json

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE FOUR DATASETS
# ============================================================================

def load_and_explore_data():
    """Load all four Simpsons datasets and explore them"""
    base_path = r"./"
    
    print("="*70)
    print("STEP 1: LOADING AND EXPLORING THE FOUR DATASETS")
    print("="*70)
    
    # Load all datasets
    characters = pd.read_csv(os.path.join(base_path, "simpsons_characters.csv"))
    print(f"\n1. CHARACTERS Dataset: {characters.shape}")
    print(f"   Columns: {list(characters.columns)}")
    
    episodes = pd.read_csv(os.path.join(base_path, "simpsons_episodes.csv"))
    print(f"\n2. EPISODES Dataset: {episodes.shape}")
    print(f"   Columns: {list(episodes.columns)}")
    
    locations = pd.read_csv(os.path.join(base_path, "simpsons_locations.csv"))
    print(f"\n3. LOCATIONS Dataset: {locations.shape}")
    print(f"   Columns: {list(locations.columns)}")
    
    script_lines = pd.read_csv(os.path.join(base_path, "simpsons_script_lines.csv"), low_memory=False)
    print(f"\n4. SCRIPT LINES Dataset: {script_lines.shape}")
    print(f"   Columns: {list(script_lines.columns)}")
    
    return {
        'episodes': episodes,
        'characters': characters,
        'locations': locations,
        'script_lines': script_lines
    }

# ============================================================================
# STEP 2: CREATE BINARY CLASSIFICATION TARGET
# ============================================================================

def create_binary_target(episodes):
    """Convert IMDB ratings to binary classification target"""
    
    print("\n" + "="*70)
    print("CREATING BINARY CLASSIFICATION TARGET")
    print("="*70)
    
    # Analyze rating distribution
    print("\n IMDB Rating Distribution:")
    print(f"   Mean: {episodes['imdb_rating'].mean():.2f}")
    print(f"   Median: {episodes['imdb_rating'].median():.2f}")
    print(f"   Min: {episodes['imdb_rating'].min():.2f}")
    print(f"   Max: {episodes['imdb_rating'].max():.2f}")
    print(f"   25th percentile: {episodes['imdb_rating'].quantile(0.25):.2f}")
    print(f"   75th percentile: {episodes['imdb_rating'].quantile(0.75):.2f}")
    
    # Use median as threshold for balanced dataset
    threshold = episodes['imdb_rating'].median()
    
    print(f"\n Classification Threshold: {threshold:.2f}")
    print(f"   Episodes with rating > {threshold:.2f} = 'Popular' (Positive class)")
    print(f"   Episodes with rating ≤ {threshold:.2f} = 'Not Popular' (Negative class)")
    
    # Create binary target
    episodes['is_popular'] = (episodes['imdb_rating'] > threshold).astype(int)
    
    # Check class distribution
    class_counts = episodes['is_popular'].value_counts()
    print(f"\n Class Distribution:")
    print(f"   Positive class (Popular): {class_counts[1]} episodes ({class_counts[1]/len(episodes)*100:.1f}%)")
    print(f"   Negative class (Not Popular): {class_counts[0]} episodes ({class_counts[0]/len(episodes)*100:.1f}%)")
    
    return episodes, threshold

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def construct_features(data):
    """Construct features predictive of episode popularity"""
    
    print("\n" + "="*70)
    print("STEP 2: CONSTRUCTING PREDICTIVE FEATURES")
    print("="*70)
    
    episodes = data['episodes'].copy()
    script = data['script_lines']
    
    # Create binary target first
    episodes, threshold = create_binary_target(episodes)
    
    # Feature 1: Time-based features
    episodes['airdate'] = pd.to_datetime(episodes['original_air_date'])
    episodes['year'] = episodes['airdate'].dt.year
    episodes['month'] = episodes['airdate'].dt.month
    episodes['day_of_week'] = episodes['airdate'].dt.dayofweek
    print("\n✓ Created time-based features: year, month, day_of_week")
    
    # Feature 2: Season and episode position
    print("✓ Using existing features: season, number_in_season")
    
    # Feature 3: Script complexity features
    lines_per_ep = script.groupby('episode_id').size().reset_index(name='total_lines')
    episodes = episodes.merge(lines_per_ep, left_on='id', right_on='episode_id', how='left')
    episodes = episodes.drop(columns=['episode_id'], errors='ignore')
    print("✓ Created feature: total_lines (script length)")
    
    chars_per_ep = script.groupby('episode_id')['character_id'].nunique().reset_index(name='unique_characters')
    episodes = episodes.merge(chars_per_ep, left_on='id', right_on='episode_id', how='left')
    episodes = episodes.drop(columns=['episode_id'], errors='ignore')
    print("✓ Created feature: unique_characters (cast size)")
    
    avg_words = script.groupby('episode_id')['word_count'].mean().reset_index(name='avg_words_per_line')
    episodes = episodes.merge(avg_words, left_on='id', right_on='episode_id', how='left')
    episodes = episodes.drop(columns=['episode_id'], errors='ignore')
    print("✓ Created feature: avg_words_per_line (dialogue complexity)")
    
    locs_per_ep = script.groupby('episode_id')['location_id'].nunique().reset_index(name='unique_locations')
    episodes = episodes.merge(locs_per_ep, left_on='id', right_on='episode_id', how='left')
    episodes = episodes.drop(columns=['episode_id'], errors='ignore')
    print("✓ Created feature: unique_locations (setting diversity)")
    
    # Feature 4: Viewership as a feature (proxy for marketing/hype)
    if 'us_viewers_in_millions' in episodes.columns:
        print("✓ Using feature: us_viewers_in_millions (viewership)")
    
    # Select final feature set
    feature_columns = [
        'season',
        'number_in_season',
        'year',
        'month',
        'day_of_week',
        'total_lines',
        'unique_characters',
        'unique_locations',
        'avg_words_per_line',
        'us_viewers_in_millions'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col in episodes.columns and episodes[col].isnull().any():
            episodes[col] = episodes[col].fillna(episodes[col].median())
    
    # Remove episodes with missing target
    episodes_clean = episodes.dropna(subset=['is_popular'])
    
    print(f"\n✓ Final dataset: {len(episodes_clean)} episodes with {len(feature_columns)} features")
    print(f"✓ Features: {feature_columns}")
    
    return episodes_clean, feature_columns, threshold

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================

def setup_train_test_split(episodes, feature_columns):
    """Setup train-test split (80/20) with stratification"""
    
    print("\n" + "="*70)
    print("STEP 3: TRAIN-TEST SPLIT SETUP")
    print("="*70)
    
    X = episodes[feature_columns]
    y = episodes['is_popular']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Training set: {len(X_train)} episodes ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Popular: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"   - Not Popular: {len(y_train)-y_train.sum()} ({(len(y_train)-y_train.sum())/len(y_train)*100:.1f}%)")
    
    print(f"\n✓ Test set: {len(X_test)} episodes ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   - Popular: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"   - Not Popular: {len(y_test)-y_test.sum()} ({(len(y_test)-y_test.sum())/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 5: TRAIN AND COMPARE CLASSIFICATION MODELS
# ============================================================================

def compare_classification_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple classification models"""
    
    print("\n" + "="*70)
    print("STEP 4: TRAINING AND COMPARING CLASSIFICATION MODELS")
    print("="*70)
    
    results = {}
    
    # Model 1: Logistic Regression
    print("\n1. Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1': f1_score(y_test, lr_pred),
        'predictions': lr_pred
    }
    print(f"   Accuracy: {results['Logistic Regression']['accuracy']:.3f}")
    print(f"   F1 Score: {results['Logistic Regression']['f1']:.3f}")
    
    # Model 2: Naive Bayes
    print("\n2. Naive Bayes")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    results['Naive Bayes'] = {
        'model': nb,
        'accuracy': accuracy_score(y_test, nb_pred),
        'precision': precision_score(y_test, nb_pred),
        'recall': recall_score(y_test, nb_pred),
        'f1': f1_score(y_test, nb_pred),
        'predictions': nb_pred
    }
    print(f"   Accuracy: {results['Naive Bayes']['accuracy']:.3f}")
    print(f"   F1 Score: {results['Naive Bayes']['f1']:.3f}")
    
    # Model 3: Decision Tree
    print("\n3. Decision Tree")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    results['Decision Tree'] = {
        'model': dt,
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred),
        'recall': recall_score(y_test, dt_pred),
        'f1': f1_score(y_test, dt_pred),
        'predictions': dt_pred
    }
    print(f"   Accuracy: {results['Decision Tree']['accuracy']:.3f}")
    print(f"   F1 Score: {results['Decision Tree']['f1']:.3f}")
    
    # Model 4: Random Forest with GridSearchCV
    print("\n4. Random Forest with GridSearchCV")
    print("   Searching hyperparameters...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    rf_pred = rf_grid.predict(X_test)
    results['Random Forest'] = {
        'model': rf_grid.best_estimator_,
        'best_params': rf_grid.best_params_,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   Accuracy: {results['Random Forest']['accuracy']:.3f}")
    print(f"   F1 Score: {results['Random Forest']['f1']:.3f}")
    
    # Model 5: Gradient Boosting with GridSearchCV
    print("\n5. Gradient Boosting with GridSearchCV")
    print("   Searching hyperparameters...")
    gb_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    gb_pred = gb_grid.predict(X_test)
    results['Gradient Boosting'] = {
        'model': gb_grid.best_estimator_,
        'best_params': gb_grid.best_params_,
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred),
        'recall': recall_score(y_test, gb_pred),
        'f1': f1_score(y_test, gb_pred),
        'predictions': gb_pred
    }
    print(f"   Best params: {gb_grid.best_params_}")
    print(f"   Accuracy: {results['Gradient Boosting']['accuracy']:.3f}")
    print(f"   F1 Score: {results['Gradient Boosting']['f1']:.3f}")
    
    # Model 6: SVM with GridSearchCV
    print("\n6. Support Vector Machine with GridSearchCV")
    print("   Searching hyperparameters...")
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(
        SVC(random_state=42),
        svm_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    svm_pred = svm_grid.predict(X_test)
    results['SVM'] = {
        'model': svm_grid.best_estimator_,
        'best_params': svm_grid.best_params_,
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1': f1_score(y_test, svm_pred),
        'predictions': svm_pred
    }
    print(f"   Best params: {svm_grid.best_params_}")
    print(f"   Accuracy: {results['SVM']['accuracy']:.3f}")
    print(f"   F1 Score: {results['SVM']['f1']:.3f}")
    
    return results

# ============================================================================
# SUMMARY AND VISUALIZATION
# ============================================================================

def print_summary(results, X_train, X_test, y_test, feature_columns, threshold):
    """Print comprehensive summary of all models"""
    
    print("\n" + "="*70)
    print("CLASSIFICATION MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nClassification Threshold: {threshold:.2f}")
    print(f"Positive Class: IMDB Rating > {threshold:.2f} (Popular)")
    print(f"Negative Class: IMDB Rating ≤ {threshold:.2f} (Not Popular)")
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<12.3f} {metrics['precision']:<12.3f} "
              f"{metrics['recall']:<12.3f} {metrics['f1']:<12.3f}")
    
    # Find best model by F1 score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model_metrics = results[best_model_name]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*70}")
    print(f"Accuracy: {best_model_metrics['accuracy']:.3f}")
    print(f"Precision: {best_model_metrics['precision']:.3f}")
    print(f"Recall: {best_model_metrics['recall']:.3f}")
    print(f"F1 Score: {best_model_metrics['f1']:.3f}")
    
    if 'best_params' in best_model_metrics:
        print(f"Best Hyperparameters: {best_model_metrics['best_params']}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_model_metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Not Popular  Predicted Popular")
    print(f"Actually Not Popular:      {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"Actually Popular:          {cm[1][0]:<15} {cm[1][1]:<15}")
    
    # Feature importance
    if hasattr(best_model_metrics['model'], 'feature_importances_'):
        print(f"\nFeature Importance (from {best_model_name}):")
        importances = best_model_metrics['model'].feature_importances_
        for feat, imp in sorted(zip(feature_columns, importances), 
                               key=lambda x: x[1], reverse=True):
            print(f"   {feat:<30} {imp:.4f}")
    
    return best_model_name, best_model_metrics

def generate_html_report(results, best_model_name, y_test, feature_columns, threshold):
    """Generate HTML report with visualizations"""
    
    best_model = results[best_model_name]['model']
    
    # Prepare data
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'name': name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    # Feature importance
    feature_importance = []
    if hasattr(best_model, 'feature_importances_'):
        for feat, imp in zip(feature_columns, best_model.feature_importances_):
            feature_importance.append({'name': feat, 'importance': float(imp)})
    
    # Confusion matrix
    cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
    
    report_data = {
        'bestModel': best_model_name,
        'threshold': float(threshold),
        'modelComparison': comparison_data,
        'featureImportance': feature_importance,
        'confusionMatrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        },
        'bestModelMetrics': {
            'accuracy': results[best_model_name]['accuracy'],
            'precision': results[best_model_name]['precision'],
            'recall': results[best_model_name]['recall'],
            'f1': results[best_model_name]['f1']
        }
    }
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simpsons Episode Popularity Classification</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #ffd700 0%, #ff6b35 100%); padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 30px; }}
        .header h1 {{ font-size: 3em; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); }}
        .card {{ background: white; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .card h2 {{ color: #ff6b35; border-bottom: 3px solid #ffd700; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #ff6b35; color: white; }}
        .best-row {{ background: #ffd70033; font-weight: bold; }}
        .chart-container {{ height: 400px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px 25px; background: #f8f9fa; border-radius: 10px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #ff6b35; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .feature-bar {{ margin: 10px 0; }}
        .bar-container {{ background: #f0f0f0; height: 30px; border-radius: 15px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: linear-gradient(90deg, #ffd700 0%, #ff6b35 100%); display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold; }}
        .confusion-matrix {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .cm-cell {{ padding: 30px; border-radius: 10px; text-align: center; font-size: 1.5em; font-weight: bold; }}
        .cm-tn {{ background: #d1fae5; color: #065f46; }}
        .cm-fp {{ background: #fee2e2; color: #991b1b; }}
        .cm-fn {{ background: #fee2e2; color: #991b1b; }}
        .cm-tp {{ background: #d1fae5; color: #065f46; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Simpsons Episode Popularity Classification</h1>
            <p>Binary Classification: Popular vs Not Popular</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Threshold: IMDB Rating > {report_data['threshold']:.2f} = Popular</p>
        </div>
        
        <div class="card">
            <h2> Best Model: {report_data['bestModel']}</h2>
            <div class="metric">
                <div class="metric-value">{report_data['bestModelMetrics']['accuracy']:.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_data['bestModelMetrics']['precision']:.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_data['bestModelMetrics']['recall']:.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_data['bestModelMetrics']['f1']:.3f}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;">
                <strong> Interpretation:</strong><br>
                The model correctly classifies <strong>{report_data['bestModelMetrics']['accuracy']*100:.1f}%</strong> of episodes.<br>
                F1 Score of <strong>{report_data['bestModelMetrics']['f1']:.3f}</strong> shows balanced precision and recall.
            </div>
        </div>
        
        <div class="card">
            <h2> Confusion Matrix</h2>
            <div class="confusion-matrix">
                <div class="cm-cell cm-tn">
                    <div>True Negatives</div>
                    <div>{report_data['confusionMatrix']['tn']}</div>
                </div>
                <div class="cm-cell cm-fp">
                    <div>False Positives</div>
                    <div>{report_data['confusionMatrix']['fp']}</div>
                </div>
                <div class="cm-cell cm-fn">
                    <div>False Negatives</div>
                    <div>{report_data['confusionMatrix']['fn']}</div>
                </div>
                <div class="cm-cell cm-tp">
                    <div>True Positives</div>
                    <div>{report_data['confusionMatrix']['tp']}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2> Model Comparison</h2>
            <div class="chart-container">
                <canvas id="comparisonChart"></canvas>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
                </thead>
                <tbody id="comparisonTable"></tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>🎯 Feature Importance</h2>
            <div id="featureImportance"></div>
        </div>
    </div>
    
    <script>
        const data = {json.dumps(report_data)};
        
        // Populate comparison table
        const tbody = document.getElementById('comparisonTable');
        data.modelComparison.forEach(model => {{
            const row = tbody.insertRow();
            if (model.name === data.bestModel) row.className = 'best-row';
            row.innerHTML = `
                <td>${{model.name}}</td>
                <td>${{model.accuracy.toFixed(3)}}</td>
                <td>${{model.precision.toFixed(3)}}</td>
                <td>${{model.recall.toFixed(3)}}</td>
                <td>${{model.f1.toFixed(3)}}</td>
            `;
        }});
        
        // Model comparison chart
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {{
                labels: data.modelComparison.map(m => m.name),
                datasets: [
                    {{
                        label: 'Accuracy',
                        data: data.modelComparison.map(m => m.accuracy),
                        backgroundColor: 'rgba(255, 215, 0, 0.7)'
                    }},
                    {{
                        label: 'F1 Score',
                        data: data.modelComparison.map(m => m.f1),
                        backgroundColor: 'rgba(255, 107, 53, 0.7)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 1 }}
                }}
            }}
        }});
        
        // Feature importance
        const featureDiv = document.getElementById('featureImportance');
        if (data.featureImportance.length > 0) {{
            data.featureImportance.forEach(feat => {{
                const pct = (feat.importance * 100).toFixed(1);
                featureDiv.innerHTML += `
                    <div class="feature-bar">
                        <div>${{feat.name}}</div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${{pct}}%">${{pct}}%</div>
                        </div>
                    </div>
                `;
            }});
        }} else {{
            featureDiv.innerHTML = '<p>Feature importance not available for this model.</p>';
        }}
    </script>
</body>
</html>'''
    
    return html

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete classification analysis pipeline"""
    
    print("\n" + "="*70)
    print("SIMPSONS EPISODE POPULARITY CLASSIFICATION")
    print("Binary Classification Task")
    print("="*70)
    
    # Step 1: Load and explore data
    data = load_and_explore_data()
    
    # Step 2: Construct features and create binary target
    episodes, feature_columns, threshold = construct_features(data)
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = setup_train_test_split(
        episodes, feature_columns
    )
    
    # Step 4: Compare classification models
    results = compare_classification_models(X_train, X_test, y_train, y_test)
    
    # Summary
    best_model_name, best_metrics = print_summary(
        results, X_train, X_test, y_test, feature_columns, threshold
    )
    
    # Generate HTML report
    html_content = generate_html_report(
        results, best_model_name, y_test, feature_columns, threshold
    )
    
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'simpsons_classification_report.html'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n{'='*70}")
    print(f"Classification analysis complete!")
    print(f"HTML report saved to: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
