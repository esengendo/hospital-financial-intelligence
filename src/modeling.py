# -*- coding: utf-8 -*-
"""
Handles the training, evaluation, and interpretation of the financial distress model.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report, fbeta_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from .config import get_config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to manage the end-to-end model training and evaluation pipeline
    with proper train/test/validation splits, imbalance handling, and feature scaling.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): DataFrame containing features and identifiers.
        """
        self.config = get_config()
        self.data = data.sort_values(by=['oshpd_id', 'year']).reset_index(drop=True)
        self.model = None
        self.best_model = None
        self.scaler = None
        self.smote = None
        self.X_train, self.X_test, self.X_val = None, None, None
        self.y_train, self.y_test, self.y_val = None, None, None
        self.X_train_scaled, self.X_test_scaled, self.X_val_scaled = None, None, None
        
        logger.info(f"📊 Initialized ModelTrainer with {len(data)} records")
        logger.info(f"📅 Year range: {data['year'].min()} - {data['year'].max()}")

    def create_target_variable(self, target_metric: str = 'operating_margin', window: int = 2, threshold: float = -5.0):
        """
        Create the binary target variable for financial distress.
        
        Distress is defined as having negative operating margin below threshold for
        `window` consecutive years. Uses available columns from our feature data.
        
        Args:
            target_metric: Which financial metric to use ('operating_margin' or 'total_margin')
            window: Number of consecutive years of poor performance to define distress
            threshold: Threshold below which performance is considered poor (default: -5.0%)
        """
        logger.info(f"Creating target variable based on {target_metric} < {threshold} for {window} consecutive years...")
        
        # Verify the target metric exists
        if target_metric not in self.data.columns:
            available_metrics = [col for col in self.data.columns if 'margin' in col.lower() and 'yoy' not in col.lower()]
            raise ValueError(f"Target metric '{target_metric}' not found. Available: {available_metrics}")
        
        self.data['is_distressed'] = 0
        
        # Ensure target metric is numeric
        self.data[target_metric] = pd.to_numeric(self.data[target_metric], errors='coerce')

        # Identify years with poor performance (below threshold)
        self.data['poor_performance'] = (self.data[target_metric] < threshold).astype(int)
        
        # Use a rolling window to check for consecutive poor performance years
        self.data['consecutive_poor'] = self.data.groupby('oshpd_id')['poor_performance'].transform(
            lambda x: x.rolling(window=window, min_periods=window).sum()
        )
        
        # Mark the first year of the distress period
        distress_indices = self.data[self.data['consecutive_poor'] == window].index
        self.data.loc[distress_indices, 'is_distressed'] = 1
        
        # Clean up temporary columns
        self.data.drop(columns=['poor_performance', 'consecutive_poor'], inplace=True)
        
        distressed_count = self.data['is_distressed'].sum()
        total_count = len(self.data)
        imbalance_ratio = distressed_count / total_count
        
        logger.info(f"✅ Target variable created: {distressed_count} distressed ({imbalance_ratio:.1%}) out of {total_count} records")
        
        if imbalance_ratio < 0.1:
            logger.warning(f"⚠️  Severe class imbalance detected ({imbalance_ratio:.1%}). Will apply SMOTE and class weighting.")
        
        return imbalance_ratio

    def split_data(self, train_end_year: int = 2019, test_end_year: int = 2020, val_start_year: int = 2021):
        """
        Split data into training, test, and out-of-time validation sets.
        
        Args:
            train_end_year: Training data up to this year (inclusive)
            test_end_year: Test data from train_end_year+1 to this year (inclusive) 
            val_start_year: Validation data from this year onwards
        """
        logger.info(f"📊 Splitting data:")
        logger.info(f"   🏋️ Training: ≤ {train_end_year}")
        logger.info(f"   🧪 Test: {train_end_year+1} - {test_end_year}")
        logger.info(f"   ✅ Validation: ≥ {val_start_year}")
        
        train_df = self.data[self.data['year'] <= train_end_year]
        test_df = self.data[(self.data['year'] > train_end_year) & (self.data['year'] <= test_end_year)]
        val_df = self.data[self.data['year'] >= val_start_year]

        # Define features (exclude identifiers and target)
        exclude_cols = ['oshpd_id', 'year', 'is_distressed', 'facility_name']
        features = [col for col in self.data.columns if col not in exclude_cols]
        
        self.X_train = train_df[features]
        self.y_train = train_df['is_distressed']
        self.X_test = test_df[features]
        self.y_test = test_df['is_distressed']
        self.X_val = val_df[features]
        self.y_val = val_df['is_distressed']
        
        # Handle missing values before scaling (required for SMOTE)
        logger.info("🧹 Handling missing values...")
        
        # Check for missing values
        missing_counts = self.X_train.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"   ⚠️  Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} features")
            
            # Check for completely missing features
            completely_missing = missing_counts[missing_counts == len(self.X_train)]
            if len(completely_missing) > 0:
                logger.info(f"   🗑️  Dropping {len(completely_missing)} completely missing features: {completely_missing.index.tolist()}")
                # Drop completely missing features from all splits
                self.X_train = self.X_train.drop(columns=completely_missing.index)
                self.X_test = self.X_test.drop(columns=completely_missing.index)
                self.X_val = self.X_val.drop(columns=completely_missing.index)
                
                # Recalculate missing counts after dropping
                missing_counts = self.X_train.isnull().sum()
            
            # Handle remaining missing values if any
            if missing_counts.sum() > 0:
                # Use SimpleImputer with median strategy for financial data
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median', fill_value=0)
                
                # Fit imputer on training data
                X_train_imputed = imputer.fit_transform(self.X_train)
                self.X_train = pd.DataFrame(
                    X_train_imputed,
                    columns=self.X_train.columns,
                    index=self.X_train.index
                )
                
                # Transform test and validation data
                X_test_imputed = imputer.transform(self.X_test)
                self.X_test = pd.DataFrame(
                    X_test_imputed,
                    columns=self.X_test.columns,
                    index=self.X_test.index
                )
                
                X_val_imputed = imputer.transform(self.X_val)
                self.X_val = pd.DataFrame(
                    X_val_imputed,
                    columns=self.X_val.columns,
                    index=self.X_val.index
                )
                
                # Store imputer for later use
                self.imputer = imputer
                logger.info(f"   ✅ Missing values imputed with median strategy for {len(self.X_train.columns)} features")
            else:
                logger.info("   ✅ No missing values remaining after dropping empty features")
                self.imputer = None
        else:
            logger.info("   ✅ No missing values found")
            self.imputer = None

        # Apply robust scaling
        logger.info("📏 Applying RobustScaler to handle financial outliers...")
        self.scaler = RobustScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        self.X_val_scaled = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )
        
        logger.info(f"✅ Data split and scaling complete:")
        logger.info(f"   🏋️ Training: {len(self.X_train)} records, {self.y_train.sum()} distressed ({self.y_train.mean():.1%})")
        logger.info(f"   🧪 Test: {len(self.X_test)} records, {self.y_test.sum()} distressed ({self.y_test.mean():.1%})")
        logger.info(f"   ✅ Validation: {len(self.X_val)} records, {self.y_val.sum()} distressed ({self.y_val.mean():.1%})")
        logger.info(f"   📊 Features: {len(features)} (scaled with RobustScaler)")

    def tune_hyperparameters(self, cv_folds: int = 3, n_jobs: int = -1, use_smote: bool = True, 
                           mode: str = 'quick', n_iter: int = None):
        """
        Perform fast hyperparameter tuning with RandomizedSearchCV.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            use_smote: Whether to use SMOTE for synthetic minority samples
            mode: 'quick' (5-10 min), 'balanced' (15-20 min), or 'thorough' (30+ min)
            n_iter: Custom number of iterations (overrides mode)
        """
        # Define mode-specific settings
        mode_settings = {
            'quick': {'n_iter': 20, 'cv_folds': 2, 'param_ranges': 'small'},
            'balanced': {'n_iter': 50, 'cv_folds': 3, 'param_ranges': 'medium'}, 
            'thorough': {'n_iter': 100, 'cv_folds': 3, 'param_ranges': 'full'}
        }
        
        if mode not in mode_settings:
            raise ValueError(f"Mode must be one of {list(mode_settings.keys())}")
        
        settings = mode_settings[mode]
        actual_n_iter = n_iter if n_iter else settings['n_iter']
        actual_cv_folds = cv_folds if cv_folds != 3 else settings['cv_folds']
        
        logger.info(f"🔧 Starting {mode} hyperparameter tuning with RandomizedSearchCV...")
        logger.info(f"   🎯 Mode: {mode} ({actual_n_iter} iterations, {actual_cv_folds}-fold CV)")
        if use_smote:
            logger.info("   🎯 Using SMOTE for synthetic minority class generation...")
        
        # Handle class imbalance - check if we have both classes
        class_counts = self.y_train.value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Training set only has one class: {class_counts.index.tolist()}. Adjust train/test split or target definition.")
        scale_pos_weight = class_counts[0] / class_counts[1]
        logger.info(f"📊 Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Define parameter distributions based on mode
        from scipy.stats import randint, uniform
        
        if settings['param_ranges'] == 'small':
            # Quick mode - focus on most impactful parameters
            param_distributions = {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 6),
                'learning_rate': uniform(0.05, 0.15),  # 0.05 to 0.20
                'subsample': uniform(0.8, 0.1),        # 0.8 to 0.9
            }
        elif settings['param_ranges'] == 'medium':
            # Balanced mode - moderate parameter space
            param_distributions = {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 7),
                'learning_rate': uniform(0.01, 0.19),  # 0.01 to 0.20
                'subsample': uniform(0.7, 0.2),        # 0.7 to 0.9
                'colsample_bytree': uniform(0.7, 0.2), # 0.7 to 0.9
                'min_child_weight': randint(1, 5)
            }
        else:  # 'full'
            # Thorough mode - full parameter space
            param_distributions = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.30
                'subsample': uniform(0.6, 0.3),        # 0.6 to 0.9
                'colsample_bytree': uniform(0.6, 0.3), # 0.6 to 0.9
                'min_child_weight': randint(1, 7),
                'gamma': uniform(0, 0.5),               # L1 regularization
                'alpha': uniform(0, 1),                 # L2 regularization
            }
        
        # Base XGBoost model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_seed,
            tree_method='hist'
        )
        
        # Create pipeline with optional SMOTE
        if use_smote:
            self.smote = SMOTE(random_state=self.config.random_seed, k_neighbors=3)
            pipeline = ImbPipeline([
                ('smote', self.smote),
                ('classifier', base_model)
            ])
            # Adjust parameter names for pipeline
            param_distributions = {f'classifier__{k}': v for k, v in param_distributions.items()}
        else:
            pipeline = base_model
        
        # Stratified cross-validation for imbalanced data
        cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=self.config.random_seed)
        
        # RandomizedSearchCV for speed
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=actual_n_iter,
            cv=cv,
            scoring='average_precision',  # PR-AUC for imbalanced data
            n_jobs=n_jobs,
            verbose=1,
            random_state=self.config.random_seed,
            return_train_score=True
        )
        
        logger.info(f"🔍 Testing {actual_n_iter} random parameter combinations...")
        estimated_time = {
            'quick': '5-10 minutes',
            'balanced': '15-20 minutes', 
            'thorough': '30+ minutes'
        }
        logger.info(f"⏱️  Estimated time: {estimated_time[mode]}")
        
        random_search.fit(self.X_train_scaled, self.y_train)
        
        self.best_model = random_search.best_estimator_
        
        logger.info("✅ Hyperparameter tuning complete!")
        logger.info(f"🏆 Best parameters: {random_search.best_params_}")
        logger.info(f"📊 Best CV score (PR-AUC): {random_search.best_score_:.4f}")
        
        return random_search.best_params_, random_search.best_score_

    def train_model(self, use_tuned_params: bool = True, use_smote: bool = True):
        """
        Train the XGBoost classifier with imbalance handling.
        
        Args:
            use_tuned_params: Whether to use hyperparameter tuned model
            use_smote: Whether to apply SMOTE to training data
        """
        if use_tuned_params and self.best_model is not None:
            logger.info("🏋️ Training final model with tuned hyperparameters...")
            self.model = self.best_model
        else:
            logger.info("🏋️ Training model with default parameters...")
            
            # Handle class imbalance - check if we have both classes
            class_counts = self.y_train.value_counts()
            if len(class_counts) < 2:
                raise ValueError(f"Training set only has one class: {class_counts.index.tolist()}. Adjust train/test split or target definition.")
            scale_pos_weight = class_counts[0] / class_counts[1]
            
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_seed,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1
            )
            
            if use_smote and self.smote is None:
                self.smote = SMOTE(random_state=self.config.random_seed, k_neighbors=3)
                self.model = ImbPipeline([
                    ('smote', self.smote),
                    ('classifier', base_model)
                ])
            else:
                self.model = base_model
        
        # Train with early stopping on test set
        if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
            self.model.fit(
                self.X_train_scaled, self.y_train,
                eval_set=[(self.X_test_scaled, self.y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(self.X_train_scaled, self.y_train)
            
        logger.info("✅ Model training complete!")

    def evaluate_model(self) -> dict:
        """
        Comprehensive model evaluation with healthcare-specific metrics.
        """
        logger.info("📊 Evaluating model performance...")
        
        results = {}
        
        for split_name, X, y in [
            ('train', self.X_train_scaled, self.y_train),
            ('test', self.X_test_scaled, self.y_test), 
            ('validation', self.X_val_scaled, self.y_val)
        ]:
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y, preds),
                'roc_auc': roc_auc_score(y, probs),
                'pr_auc': average_precision_score(y, probs),
                'f1_score': f1_score(y, preds),
                'f2_score': fbeta_score(y, preds, beta=2),  # Emphasizes recall for healthcare
                'precision': classification_report(y, preds, output_dict=True)['1']['precision'] if '1' in classification_report(y, preds, output_dict=True) else 0,
                'recall': classification_report(y, preds, output_dict=True)['1']['recall'] if '1' in classification_report(y, preds, output_dict=True) else 0,
                'support': int(y.sum())
            }
            
            results[split_name] = metrics
            
            logger.info(f"📊 {split_name.upper()} Performance:")
            logger.info(f"   🎯 Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"   📈 ROC-AUC: {metrics['roc_auc']:.3f}")
            logger.info(f"   📊 PR-AUC: {metrics['pr_auc']:.3f}")
            logger.info(f"   🎯 F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"   🎯 F2-Score: {metrics['f2_score']:.3f} (recall-focused)")
            logger.info(f"   📋 Support: {metrics['support']} positive cases")
        
        return results

    def save_model(self, model_name: str = None) -> Path:
        """Save the trained model, scaler, and metadata."""
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"hospital_distress_model_{timestamp}"
        
        model_dir = self.config.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save imputer (if used)
        if hasattr(self, 'imputer') and self.imputer is not None:
            imputer_path = model_dir / "imputer.pkl"
            joblib.dump(self.imputer, imputer_path)
        
        # Save feature names
        features_path = model_dir / "features.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.X_train.columns))
        
        # Save model metadata
        metadata = {
            'model_type': 'XGBoost with SMOTE and RobustScaler',
            'features_count': len(self.X_train.columns),
            'training_records': len(self.X_train),
            'class_distribution': self.y_train.value_counts().to_dict(),
            'scaling_method': 'RobustScaler',
            'imbalance_handling': 'SMOTE + class_weight',
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {model_dir}")
        logger.info(f"   📦 Model: {model_path}")
        logger.info(f"   📏 Scaler: {scaler_path}")
        logger.info(f"   📋 Metadata: {metadata_path}")
        
        return model_dir

    def plot_feature_importance(self, output_dir: Path = None, top_n: int = 20):
        """
        Create and save comprehensive feature importance plots.
        """
        if output_dir is None:
            output_dir = self.config.shap_outputs_dir
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"📊 Generating feature importance plots (top {top_n})...")
        
        # Get the actual classifier from pipeline if needed
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps['classifier']
        else:
            classifier = self.model
        
        # SHAP feature importance
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(self.X_val_scaled.iloc[:100])  # Sample for speed
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_val_scaled.iloc[:100], show=False, plot_type="bar", max_display=top_n)
        plt.title("SHAP Feature Importance - Hospital Financial Distress")
        plt.tight_layout()
        
        shap_path = output_dir / "shap_feature_importance.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Built-in XGBoost feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(classifier, max_num_features=top_n, importance_type='weight')
        plt.title("XGBoost Feature Importance (Weight) - Hospital Financial Distress")
        plt.tight_layout()
        
        xgb_path = output_dir / "xgboost_feature_importance.png"
        plt.savefig(xgb_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Feature importance plots saved:")
        logger.info(f"   📊 SHAP: {shap_path}")
        logger.info(f"   📊 XGBoost: {xgb_path}")
        
        return shap_path, xgb_path 