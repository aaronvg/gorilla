#!/usr/bin/env python3
"""
BFCL Results Visualization Tool

A Streamlit app to visualize Berkeley Function Call Leaderboard results
across different models and tool formats.

Usage:
    uv run --with streamlit --with plotly --with pandas streamlit run visualize_bfcl_results.py

Or with dependencies installed:
    pip install streamlit plotly pandas
    streamlit run visualize_bfcl_results.py
"""

import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any


@st.cache_data
def load_all_data(score_dir: str = "score", result_dir: str = "result") -> Dict[str, Any]:
    """Load all data from score and result directories."""
    score_path = Path(score_dir)
    result_path = Path(result_dir)
    
    data = {
        "models": {},
        "csv_data": {},
        "results": {},
        "test_data": {}
    }
    
    # Load test data for questions
    test_data_path = Path("bfcl_eval/data")
    if test_data_path.exists():
        for test_file in test_data_path.glob("BFCL_v3_*.json"):
            category = test_file.stem.replace("BFCL_v3_", "")
            try:
                with open(test_file, 'r') as f:
                    tests = []
                    for line in f:
                        if line.strip():
                            tests.append(json.loads(line))
                    data["test_data"][category] = tests
            except Exception as e:
                st.warning(f"Could not load test data {test_file}: {e}")
    
    # Load result data (actual model outputs)
    if result_path.exists():
        for model_dir in result_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                data["results"][model_name] = {}
                
                for result_file in model_dir.glob("*.json"):
                    category = result_file.stem.replace("BFCL_v3_", "").replace("_result", "")
                    try:
                        with open(result_file, 'r') as f:
                            results = []
                            for line in f:
                                if line.strip():
                                    results.append(json.loads(line))
                            data["results"][model_name][category] = results
                    except Exception as e:
                        st.warning(f"Could not load result file {result_file}: {e}")
    
    # Load score data  
    if not score_path.exists():
        st.error(f"Score directory '{score_dir}' not found!")
        return data
    
    # Load CSV files
    csv_files = ["data_overall.csv", "data_non_live.csv", "data_live.csv", "data_multi_turn.csv"]
    for csv_file in csv_files:
        csv_path = score_path / csv_file
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                data["csv_data"][csv_file.replace(".csv", "")] = df
            except Exception as e:
                st.warning(f"Could not load {csv_file}: {e}")
    
    # Load individual model score files
    for model_dir in score_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            data["models"][model_name] = {}
            
            for score_file in model_dir.glob("*.json"):
                test_category = score_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                try:
                    with open(score_file, 'r') as f:
                        score_data = []
                        for line in f:
                            if line.strip():
                                score_data.append(json.loads(line))
                        data["models"][model_name][test_category] = score_data
                except Exception as e:
                    st.warning(f"Could not load {score_file}: {e}")
    
    return data


def extract_model_summary(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract summary statistics for each model."""
    summaries = []
    
    for model_name, categories in data["models"].items():
        for category, results in categories.items():
            if not results:
                continue
                
            # First item contains overall accuracy
            overall_result = results[0]
            if "accuracy" in overall_result:
                summary = {
                    "Model": model_name,
                    "Category": category,
                    "Accuracy": overall_result["accuracy"],
                    "Correct": overall_result.get("correct_count", 0),
                    "Total": overall_result.get("total_count", 0),
                    "Tool Format": "BAML" if "-baml" in model_name else "FC" if "-FC" in model_name else "Prompt"
                }
                summaries.append(summary)
    
    return pd.DataFrame(summaries)


def analyze_error_patterns(data: Dict[str, Any]) -> pd.DataFrame:
    """Analyze error patterns across models."""
    error_data = []
    
    for model_name, categories in data["models"].items():
        for category, results in categories.items():
            if not results:
                continue
                
            # Skip the first item (overall stats) and analyze individual test results
            for result in results[1:]:
                if "error_type" in result:
                    error_data.append({
                        "Model": model_name,
                        "Category": category,
                        "Error Type": result["error_type"],
                        "Valid": result.get("valid", False),
                        "Test ID": result.get("id", ""),
                        "Tool Format": "BAML" if "-baml" in model_name else "FC" if "-FC" in model_name else "Prompt"
                    })
    
    return pd.DataFrame(error_data)


def create_accuracy_comparison(summary_df: pd.DataFrame):
    """Create accuracy comparison chart."""
    fig = px.bar(
        summary_df,
        x="Model",
        y="Accuracy",
        color="Tool Format",
        facet_col="Category",
        title="Model Accuracy by Category and Tool Format",
        labels={"Accuracy": "Accuracy (%)", "Model": "Model"},
        color_discrete_map={
            "BAML": "#FF6B6B",
            "FC": "#4ECDC4", 
            "Prompt": "#45B7D1"
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_tickangle=-45
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1%")
    
    return fig


def create_error_analysis(error_df: pd.DataFrame):
    """Create error pattern analysis."""
    if error_df.empty:
        return None
        
    # Count errors by type and model
    error_counts = error_df.groupby(["Model", "Tool Format", "Error Type"]).size().reset_index(name="Count")
    
    fig = px.bar(
        error_counts,
        x="Model",
        y="Count",
        color="Error Type",
        facet_col="Tool Format",
        title="Error Patterns by Model and Tool Format",
        labels={"Count": "Number of Errors", "Model": "Model"}
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_tickangle=-45
    )
    
    return fig


def create_detailed_comparison(summary_df: pd.DataFrame):
    """Create detailed comparison table."""
    # Pivot to compare tool formats side by side
    pivot_df = summary_df.pivot_table(
        index=["Model", "Category"],
        columns="Tool Format",
        values="Accuracy",
        aggfunc="first"
    ).reset_index()
    
    # Fill NaN with 0 and format as percentages
    for col in pivot_df.columns:
        if col not in ["Model", "Category"]:
            pivot_df[col] = pivot_df[col].fillna(0)
    
    return pivot_df


def main():
    st.set_page_config(
        page_title="BFCL Results Visualization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🦍 Berkeley Function Call Leaderboard (BFCL) Results")
    st.markdown("Visualizing model performance across different tool formats")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    score_dir = st.sidebar.text_input("Score Directory", value="score")
    
    # Load data
    result_dir = st.sidebar.text_input("Result Directory", value="result")
    
    with st.spinner("Loading data..."):
        data = load_all_data(score_dir, result_dir)
    
    if not data["models"]:
        st.error("No model data found! Make sure the score directory exists and contains model results.")
        return
    
    # Extract summary statistics
    summary_df = extract_model_summary(data)
    error_df = analyze_error_patterns(data)
    
    if summary_df.empty:
        st.error("No summary data could be extracted from the results.")
        return
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", len(summary_df["Model"].unique()))
    
    with col2:
        st.metric("Categories Tested", len(summary_df["Category"].unique()))
    
    with col3:
        avg_accuracy = summary_df["Accuracy"].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
    
    # Accuracy comparison
    st.header("📈 Accuracy Comparison")
    accuracy_fig = create_accuracy_comparison(summary_df)
    st.plotly_chart(accuracy_fig, use_container_width=True)
    
    # Tool format comparison
    st.header("🔧 Tool Format Analysis")
    
    tool_format_summary = summary_df.groupby("Tool Format").agg({
        "Accuracy": ["mean", "count"],
        "Correct": "sum",
        "Total": "sum"
    }).round(4)
    
    tool_format_summary.columns = ["Mean Accuracy", "Test Count", "Total Correct", "Total Tests"]
    tool_format_summary["Overall Accuracy"] = tool_format_summary["Total Correct"] / tool_format_summary["Total Tests"]
    
    st.dataframe(tool_format_summary, use_container_width=True)
    
    # Error analysis
    if not error_df.empty:
        st.header("❌ Error Analysis")
        error_fig = create_error_analysis(error_df)
        if error_fig:
            st.plotly_chart(error_fig, use_container_width=True)
        
        # Error type breakdown
        st.subheader("Error Type Breakdown")
        error_summary = error_df.groupby(["Tool Format", "Error Type"]).size().reset_index(name="Count")
        error_pivot = error_summary.pivot(index="Error Type", columns="Tool Format", values="Count").fillna(0)
        st.dataframe(error_pivot, use_container_width=True)
    
    # Detailed results table
    st.header("📋 Detailed Results")
    detailed_df = create_detailed_comparison(summary_df)
    st.dataframe(detailed_df, use_container_width=True)
    
    # Question-by-question comparison
    st.header("🔎 Question-by-Question Analysis")
    
    if data["test_data"] and data["results"]:
        # Category selection
        available_categories = list(data["test_data"].keys())
        selected_category = st.selectbox("Select Test Category", options=available_categories)
        
        if selected_category and selected_category in data["test_data"]:
            test_questions = data["test_data"][selected_category]
            
            # Model selection for comparison
            available_models = [m for m in data["results"].keys() if selected_category in data["results"][m]]
            selected_models = st.multiselect(
                "Select Models to Compare", 
                options=available_models,
                default=available_models[:3] if len(available_models) >= 3 else available_models
            )
            
            if selected_models and test_questions:
                # Question selection
                question_options = {f"{q['id']}: {q['question'][0][0]['content'][:100]}...": q['id'] 
                                  for q in test_questions}
                selected_question_display = st.selectbox("Select Question", options=list(question_options.keys()))
                selected_question_id = question_options[selected_question_display]
                
                # Find the selected question details
                question_data = next(q for q in test_questions if q['id'] == selected_question_id)
                
                # Display question details
                st.subheader(f"Question: {selected_question_id}")
                st.write("**User Query:**")
                st.info(question_data['question'][0][0]['content'])
                
                st.write("**Available Functions:**")
                if 'function' in question_data and question_data['function']:
                    for func in question_data['function']:
                        func_name = func.get('name', 'Unknown Function')
                        with st.expander(f"Function: {func_name}"):
                            st.json(func)
                else:
                    st.write("No functions defined for this question")
                
                # Compare model responses
                st.subheader("Model Responses Comparison")
                
                cols = st.columns(len(selected_models))
                
                for i, model in enumerate(selected_models):
                    with cols[i]:
                        st.write(f"**{model}**")
                        
                        # Find this model's response to this question
                        model_results = data["results"][model].get(selected_category, [])
                        question_result = next((r for r in model_results if r['id'] == selected_question_id), None)
                        
                        if question_result:
                            # Show the result
                            st.write("*Raw Result:*")
                            st.code(json.dumps(question_result['result'], indent=2), language='json')
                            
                            # Show metadata
                            metadata = {k: v for k, v in question_result.items() 
                                      if k not in ['id', 'result'] and not k.startswith('_')}
                            if metadata:
                                with st.expander("Metadata"):
                                    st.json(metadata)
                        else:
                            st.error("No result found for this question")
                        
                        # Show score if available
                        if selected_category in data["models"].get(model, {}):
                            score_results = data["models"][model][selected_category]
                            question_score = next((s for s in score_results[1:] if s.get('id') == selected_question_id), None)
                            
                            if question_score:
                                if question_score.get('valid', False):
                                    st.success("✅ Correct")
                                else:
                                    st.error("❌ Incorrect")
                                    if 'error' in question_score:
                                        st.write("*Errors:*")
                                        for error in question_score['error']:
                                            st.write(f"- {error}")
                                    if 'error_type' in question_score:
                                        st.write(f"*Error Type:* {question_score['error_type']}")
    
    # Batch comparison
    st.header("📊 Batch Question Analysis")
    
    if data["test_data"] and data["results"]:
        # Create a comprehensive comparison table
        comparison_data = []
        
        for category, questions in data["test_data"].items():
            for question in questions:
                try:
                    question_id = question.get('id', 'unknown')
                    # Safely extract question text
                    question_text = "N/A"
                    if 'question' in question and question['question'] and len(question['question']) > 0:
                        if len(question['question'][0]) > 0 and 'content' in question['question'][0][0]:
                            question_text = question['question'][0][0]['content']
                
                    # Get function name safely
                    function_name = 'N/A'
                    if 'function' in question and question['function']:
                        function_name = question['function'][0].get('name', 'N/A')
                    
                    row = {
                        'Category': category,
                        'Question ID': question_id,
                        'Question': question_text[:150] + '...' if len(question_text) > 150 else question_text,
                        'Function': function_name
                    }
                    
                    # Add results from each model
                    for model_name in data["results"].keys():
                        if category in data["results"][model_name]:
                            model_results = data["results"][model_name][category]
                            question_result = next((r for r in model_results if r['id'] == question_id), None)
                            
                            # Get correctness from score data
                            is_correct = "N/A"
                            if category in data["models"].get(model_name, {}):
                                score_results = data["models"][model_name][category]
                                question_score = next((s for s in score_results[1:] if s.get('id') == question_id), None)
                                if question_score:
                                    is_correct = "✅" if question_score.get('valid', False) else "❌"
                            
                            # Format the result
                            if question_result:
                                result_str = str(question_result['result'])
                                if len(result_str) > 100:
                                    result_str = result_str[:100] + "..."
                                row[f'{model_name} Result'] = result_str
                                row[f'{model_name} Correct'] = is_correct
                            else:
                                row[f'{model_name} Result'] = "No result"
                                row[f'{model_name} Correct'] = "N/A"
                    
                    comparison_data.append(row)
                
                except Exception as e:
                    st.warning(f"Error processing question {question.get('id', 'unknown')} in category {category}: {e}")
                    continue
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Filter options
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=comparison_df['Category'].unique(),
                    default=comparison_df['Category'].unique()
                )
            
            with col2:
                # Show only failed cases option
                show_only_failures = st.checkbox("Show only failed cases")
            
            # Apply filters
            filtered_df = comparison_df[comparison_df['Category'].isin(category_filter)]
            
            if show_only_failures:
                # Filter to show only rows where at least one model failed
                correct_columns = [col for col in filtered_df.columns if col.endswith(' Correct')]
                if correct_columns:
                    # Keep rows where any model has ❌
                    mask = filtered_df[correct_columns].apply(lambda row: '❌' in row.values, axis=1)
                    filtered_df = filtered_df[mask]
            
            st.subheader("Comparison Table")
            st.dataframe(filtered_df, use_container_width=True, height=600)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="bfcl_comparison.csv",
                mime="text/csv"
            )
    
    # Raw data explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.subheader("Model Selection")
        selected_model = st.selectbox("Select Model", options=list(data["models"].keys()))
        
        if selected_model:
            st.subheader(f"Results for {selected_model}")
            model_data = data["models"][selected_model]
            
            for category, results in model_data.items():
                if results:
                    st.write(f"**{category.title()}**")
                    overall_stats = results[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{overall_stats.get('accuracy', 0):.1%}")
                    with col2:
                        st.metric("Correct", overall_stats.get('correct_count', 0))
                    with col3:
                        st.metric("Total", overall_stats.get('total_count', 0))
                    
                    # Show failed cases
                    failed_cases = [r for r in results[1:] if not r.get('valid', True)]
                    if failed_cases:
                        st.write(f"Failed cases: {len(failed_cases)}")
                        with st.expander(f"Show failed cases for {category}"):
                            for case in failed_cases[:10]:  # Limit to first 10
                                st.json(case)
    
    # CSV data explorer
    if data["csv_data"]:
        with st.expander("📊 CSV Data"):
            for csv_name, df in data["csv_data"].items():
                st.subheader(f"{csv_name.replace('_', ' ').title()}")
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()