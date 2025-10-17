"""
Master Script - Run Complete Analysis Pipeline
==============================================
This script runs the entire analysis pipeline in order:
1. Data exploration and preparation
2. Feature selection analysis
3. All model training (6 estimators)
4. Final comparison and test evaluation

WARNING: This may take several minutes to complete!
"""

import sys


def run_step(step_name, script_name):
    """Run a script and handle errors"""
    print("\n" + "="*80)
    print(f"STEP: {step_name}")
    print("="*80)
    
    try:
        if script_name == "data_preparation":
            from data_preparation import load_and_explore_data, visualize_data, prepare_data
            df = load_and_explore_data()
            visualize_data(df, save_plots=True)
            prepare_data()
            
        elif script_name == "feature_selection":
            from feature_selection import main as fs_main
            fs_main()
            
        elif script_name == "final_comparison":
            from final_comparison import main as fc_main
            fc_main()
            
        print(f"\n✓ {step_name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in {step_name}:")
        print(f"  {str(e)}")
        return False


def main():
    """Run complete pipeline"""
    print("="*80)
    print("SPOTIFY CHURN PREDICTION - COMPLETE ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis will run all analysis steps:")
    print("  1. Data Exploration & Preparation")
    print("  2. Feature Selection Analysis")
    print("  3. Model Training (all 6 estimators)")
    print("  4. Final Comparison & Test Evaluation")
    print("\nEstimated time: 5-10 minutes")
    print("="*80)
    
    # Step 1: Data Preparation
    if not run_step("Data Exploration & Preparation", "data_preparation"):
        print("\nStopping pipeline due to error.")
        return
    
    # Step 2: Feature Selection
    if not run_step("Feature Selection Analysis", "feature_selection"):
        print("\nStopping pipeline due to error.")
        return
    
    # Step 3 & 4: Final Comparison (includes all model training)
    if not run_step("Model Training & Comparison", "final_comparison"):
        print("\nStopping pipeline due to error.")
        return
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 Visualization plots (PNG files)")
    print("  📈 Learning curves for all 6 models")
    print("  📉 Model comparison charts")
    print("  📋 Test set evaluation results")
    print("\nNext steps:")
    print("  1. Review all generated plots")
    print("  2. Prepare presentation slides (8-10 minutes)")
    print("  3. Include key findings and best model results")
    print("="*80)


if __name__ == "__main__":
    main()

