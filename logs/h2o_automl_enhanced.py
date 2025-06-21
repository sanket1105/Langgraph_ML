
def h2o_automl_enhanced(train_data, test_data, calib_data, target_variable, feature_columns, enable_optuna=True, optuna_n_trials=50, optuna_timeout=300, model_directory=None, log_path=None, enable_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name="H2O AutoML Enhanced", mlflow_run_name=None, **kwargs):
    import h2o
    from h2o.automl import H2OAutoML
    import pandas as pd
    import numpy as np
    import mapie
    from contextlib import nullcontext
    
    # Optional imports
    if enable_optuna:
        import optuna
        from optuna.samplers import TPESampler
    
    if enable_mlflow:
        import mlflow
        import mlflow.h2o
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        run_context = mlflow.start_run(run_name=mlflow_run_name)
    else:
        run_context = nullcontext()

    # Convert data to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    calib_df = pd.DataFrame(calib_data)

    with run_context as run:
        # Initialize H2O
        h2o.init()

        # Create H2OFrames
        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
        calib_h2o = h2o.H2OFrame(calib_df)

        # Convert target variable to categorical if it's binary
        # Check if target has only 2 unique values by converting to pandas first
        target_values = train_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
        if len(set(target_values)) == 2:
            train_h2o[target_variable] = train_h2o[target_variable].asfactor()
            test_h2o[target_variable] = test_h2o[target_variable].asfactor()
            calib_h2o[target_variable] = calib_h2o[target_variable].asfactor()

        # Train AutoML model
        aml = H2OAutoML(
            max_runtime_secs=300,
            max_models=20,
            nfolds=5,
            seed=42,
            sort_metric="AUTO"
        )
        
        aml.train(x=feature_columns, y=target_variable, training_frame=train_h2o)
        
        # Evaluate on test set
        test_perf = aml.leader.model_performance(test_h2o)
        test_metrics = {}
        
        # Handle classification metrics
        try:
            if hasattr(test_perf, 'auc'):
                auc_value = test_perf.auc()
                test_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
        except:
            pass
            
        try:
            if hasattr(test_perf, 'logloss'):
                logloss_value = test_perf.logloss()
                test_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
        except:
            pass
            
        # Calculate Brier Score for binary classification
        try:
            if len(set(target_values)) == 2:  # Binary classification
                # Get predicted probabilities
                test_pred = aml.leader.predict(test_h2o)
                test_probs = test_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                test_actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                
                # Convert to numeric if categorical
                if test_actual.dtype == 'object':
                    test_actual = (test_actual == test_actual[0]).astype(int)
                
                # Calculate Brier Score
                brier_score = np.mean((test_probs - test_actual) ** 2)
                test_metrics['brier_score'] = brier_score
        except Exception as e:
            print(f"Could not calculate Brier score: {{e}}")
            
        # Handle regression metrics
        try:
            if hasattr(test_perf, 'rmse'):
                rmse_value = test_perf.rmse()
                test_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
        except:
            pass
            
        try:
            if hasattr(test_perf, 'mae'):
                mae_value = test_perf.mae()
                test_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
        except:
            pass
        
        # Evaluate on calibration set
        calib_perf = aml.leader.model_performance(calib_h2o)
        calib_metrics = {}
        
        # Handle classification metrics
        try:
            if hasattr(calib_perf, 'auc'):
                auc_value = calib_perf.auc()
                calib_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
        except:
            pass
            
        try:
            if hasattr(calib_perf, 'logloss'):
                logloss_value = calib_perf.logloss()
                calib_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
        except:
            pass
            
        # Calculate Brier Score for calibration set
        try:
            if len(set(target_values)) == 2:  # Binary classification
                # Get predicted probabilities
                calib_pred = aml.leader.predict(calib_h2o)
                calib_probs = calib_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                calib_actual = calib_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                
                # Convert to numeric if categorical
                if calib_actual.dtype == 'object':
                    calib_actual = (calib_actual == calib_actual[0]).astype(int)
                
                # Calculate Brier Score
                brier_score = np.mean((calib_probs - calib_actual) ** 2)
                calib_metrics['brier_score'] = brier_score
        except Exception as e:
            print(f"Could not calculate Brier score for calibration set: {{e}}")
            
        # Handle regression metrics
        try:
            if hasattr(calib_perf, 'rmse'):
                rmse_value = calib_perf.rmse()
                calib_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
        except:
            pass
            
        try:
            if hasattr(calib_perf, 'mae'):
                mae_value = calib_perf.mae()
                calib_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
        except:
            pass

        # Save model if directory provided
        model_path = None
        if model_directory or log_path:
            save_path = model_directory if model_directory else log_path
            model_path = h2o.save_model(model=aml.leader, path=save_path, force=True)

        # Get leaderboard
        leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
        # Compute Brier Score for all models in the leaderboard (binary classification only)
        if len(set(target_values)) == 2:
            brier_scores = []
            for model_id in leaderboard_df['model_id']:
                model = h2o.get_model(model_id)
                pred = model.predict(test_h2o)
                if 'p1' in pred.columns:
                    probs = pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()
                else:
                    # fallback to first probability column if p1 not present
                    prob_cols = [col for col in pred.columns if col.startswith('p')]
                    probs = pred[prob_cols[0]].as_data_frame(use_multi_thread=True).values.flatten()
                actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                if actual.dtype == 'object':
                    actual = (actual == actual[0]).astype(int)
                brier = np.mean((probs - actual) ** 2)
                brier_scores.append(brier)
            leaderboard_df['brier_score'] = brier_scores
            leaderboard_df = leaderboard_df.sort_values('brier_score', ascending=True).reset_index(drop=True)
        leaderboard_dict = leaderboard_df.to_dict()

        # Set best model as the one with the lowest Brier Score (if available)
        if 'brier_score' in leaderboard_df.columns:
            best_model_id = leaderboard_df.loc[0, 'model_id']
            best_model = h2o.get_model(best_model_id)
            if model_directory or log_path:
                save_path = model_directory if model_directory else log_path
                model_path = h2o.save_model(model=best_model, path=save_path, force=True)
            else:
                model_path = None
        else:
            best_model_id = aml.leader.model_id
            model_path = model_path if 'model_path' in locals() else None

        # Get model type and parameters
        model_type = type(best_model).__name__
        model_params = best_model.params if hasattr(best_model, "params") else {}

        # Prepare results
        results = {
            'leaderboard': leaderboard_dict,
            'best_model_id': best_model_id,
            'model_path': model_path,
            'test_metrics': test_metrics,
            'calibration_metrics': calib_metrics,
            'optimization_results': None,
            'model_results': {
                'model_flavor': 'H2O AutoML Enhanced',
                'model_path': model_path,
                'best_model_id': best_model_id,
                'test_performance': test_metrics,
                'calibration_performance': calib_metrics
            },
            'model_type': model_type,
            'model_parameters': model_params
        }

        # Add CDF vs EDF PDF for all models
        try:
            cdf_edf_pdf_path = plot_cdf_vs_edf_for_models(leaderboard_df, test_h2o, target_variable, logs_dir=log_path or "logs/")
            results['cdf_edf_pdf_path'] = cdf_edf_pdf_path
        except Exception as e:
            print(f"Could not generate CDF vs EDF PDF: {e}")
            results['cdf_edf_pdf_path'] = None

        return results
