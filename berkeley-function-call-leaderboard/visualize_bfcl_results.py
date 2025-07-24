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
def load_all_data(
    score_dir: str = "score", result_dir: str = "result"
) -> Dict[str, Any]:
    """Load all data from score and result directories."""
    score_path = Path(score_dir)
    result_path = Path(result_dir)

    data = {"models": {}, "csv_data": {}, "results": {}, "test_data": {}}

    # Load test data for questions
    test_data_path = Path("bfcl_eval/data")
    if test_data_path.exists():
        for test_file in test_data_path.glob("BFCL_v3_*.json"):
            category = test_file.stem.replace("BFCL_v3_", "")
            try:
                with open(test_file, "r") as f:
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
                    category = result_file.stem.replace("BFCL_v3_", "").replace(
                        "_result", ""
                    )
                    try:
                        with open(result_file, "r") as f:
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
    csv_files = [
        "data_overall.csv",
        "data_non_live.csv",
        "data_live.csv",
        "data_multi_turn.csv",
    ]
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
                test_category = score_file.stem.replace("BFCL_v3_", "").replace(
                    "_score", ""
                )
                try:
                    with open(score_file, "r") as f:
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
                    "Tool Format": (
                        "BAML"
                        if "-baml" in model_name
                        else "FC" if "-FC" in model_name else "Prompt"
                    ),
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
                    error_data.append(
                        {
                            "Model": model_name,
                            "Category": category,
                            "Error Type": result["error_type"],
                            "Valid": result.get("valid", False),
                            "Test ID": result.get("id", ""),
                            "Tool Format": (
                                "BAML"
                                if "-baml" in model_name
                                else "FC" if "-FC" in model_name else "Prompt"
                            ),
                        }
                    )

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
        color_discrete_map={"BAML": "#FF6B6B", "FC": "#4ECDC4", "Prompt": "#45B7D1"},
    )

    fig.update_layout(height=500, showlegend=True, xaxis_tickangle=-45)

    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1%")

    return fig


def create_error_analysis(error_df: pd.DataFrame):
    """Create error pattern analysis."""
    if error_df.empty:
        return None

    # Count errors by type and model
    error_counts = (
        error_df.groupby(["Model", "Tool Format", "Error Type"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        error_counts,
        x="Model",
        y="Count",
        color="Error Type",
        facet_col="Tool Format",
        title="Error Patterns by Model and Tool Format",
        labels={"Count": "Number of Errors", "Model": "Model"},
    )

    fig.update_layout(height=500, showlegend=True, xaxis_tickangle=-45)

    return fig


def create_detailed_comparison(summary_df: pd.DataFrame):
    """Create detailed comparison table."""
    # Pivot to compare tool formats side by side
    pivot_df = summary_df.pivot_table(
        index=["Model", "Category"],
        columns="Tool Format",
        values="Accuracy",
        aggfunc="first",
    ).reset_index()

    # Fill NaN with 0 and format as percentages
    for col in pivot_df.columns:
        if col not in ["Model", "Category"]:
            pivot_df[col] = pivot_df[col].fillna(0)

    return pivot_df


def main():
    st.set_page_config(
        page_title="BFCL Results Visualization", page_icon="📊", layout="wide"
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
        st.error(
            "No model data found! Make sure the score directory exists and contains model results."
        )
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

    tool_format_summary = (
        summary_df.groupby("Tool Format")
        .agg({"Accuracy": ["mean", "count"], "Correct": "sum", "Total": "sum"})
        .round(4)
    )

    tool_format_summary.columns = [
        "Mean Accuracy",
        "Test Count",
        "Total Correct",
        "Total Tests",
    ]
    tool_format_summary["Overall Accuracy"] = (
        tool_format_summary["Total Correct"] / tool_format_summary["Total Tests"]
    )

    st.dataframe(tool_format_summary, use_container_width=True)

    # Error analysis
    if not error_df.empty:
        st.header("❌ Error Analysis")
        error_fig = create_error_analysis(error_df)
        if error_fig:
            st.plotly_chart(error_fig, use_container_width=True)

        # Error type breakdown
        st.subheader("Error Type Breakdown")
        error_summary = (
            error_df.groupby(["Tool Format", "Error Type"])
            .size()
            .reset_index(name="Count")
        )
        error_pivot = error_summary.pivot(
            index="Error Type", columns="Tool Format", values="Count"
        ).fillna(0)
        st.dataframe(error_pivot, use_container_width=True)

    # Detailed results table
    st.header("📋 Detailed Results")
    detailed_df = create_detailed_comparison(summary_df)
    st.dataframe(detailed_df, use_container_width=True)

    # CSV data explorer
    if data["csv_data"]:
        with st.expander("📊 CSV Data"):
            for csv_name, df in data["csv_data"].items():
                st.subheader(f"{csv_name.replace('_', ' ').title()}")
                st.dataframe(df, use_container_width=True)

    # Deep Dive section
    st.header("🔍 Deep Dive")
    
    # Create two columns for the deep dive section
    col_left, col_right = st.columns([1, 2])
    
    # Collect all questions from test data with result status
    all_questions = []
    for category, tests in data["test_data"].items():
        for test in tests:
            question_id = test.get("id", "")
            
            # Check if any results exist and if there are failures
            has_data = False
            has_baml_failure = False
            has_fc_failure = False
            
            for model_name in data["results"]:
                if category in data["results"][model_name]:
                    results = data["results"][model_name][category]
                    for result in results:
                        if result.get("id") == question_id:
                            has_data = True
                            
                            # Check for failures in score data
                            if model_name in data["models"] and category in data["models"][model_name]:
                                scores = data["models"][model_name][category]
                                for score in scores[1:]:  # Skip overall stats
                                    if score.get("id") == question_id:
                                        is_valid = score.get("valid", False)
                                        if not is_valid:
                                            if "-baml" in model_name:
                                                has_baml_failure = True
                                            elif "-FC" in model_name:
                                                has_fc_failure = True
                                        break
                            break
            
            all_questions.append({
                "id": question_id,
                "category": category,
                "question": test,
                "has_data": has_data,
                "has_baml_failure": has_baml_failure,
                "has_fc_failure": has_fc_failure
            })
    
    if all_questions:
        # Left column: Question list
        with col_left:
            st.subheader("Questions")
            
            # Add filters
            st.write("**Filters:**")
            
            # Category filter
            categories = ["All"] + list(data["test_data"].keys())
            selected_category = st.selectbox("Category:", categories)
            
            # Data presence filter
            filter_has_data = st.checkbox("Only questions with results", value=True)
            
            # Failure filters
            filter_baml_failures = st.checkbox("Only BAML failures")
            filter_fc_failures = st.checkbox("Only FC failures")
            
            # Apply filters
            filtered_questions = all_questions
            
            if selected_category != "All":
                filtered_questions = [q for q in filtered_questions if q["category"] == selected_category]
            
            if filter_has_data:
                filtered_questions = [q for q in filtered_questions if q["has_data"]]
                
            if filter_baml_failures:
                filtered_questions = [q for q in filtered_questions if q["has_baml_failure"]]
                
            if filter_fc_failures:
                filtered_questions = [q for q in filtered_questions if q["has_fc_failure"]]
            
            # Create scrollable list of questions
            st.write(f"**Questions ({len(filtered_questions)}):**")
            
            if filtered_questions:
                question_display = []
                for q in filtered_questions:
                    status_indicators = []
                    if q["has_baml_failure"]:
                        status_indicators.append("❌BAML")
                    if q["has_fc_failure"]:
                        status_indicators.append("❌FC")
                    status_str = " ".join(status_indicators) if status_indicators else ""
                    display_text = f"{q['id']} ({q['category']})"
                    if status_str:
                        display_text += f" {status_str}"
                    question_display.append(display_text)
                
                selected_question_idx = st.selectbox(
                    "Select a question:",
                    range(len(filtered_questions)),
                    format_func=lambda x: question_display[x],
                    key="question_selector"
                )
            else:
                st.info("No questions match the selected filters.")
            
        # Right column: Question details and results
        with col_right:
            if filtered_questions and selected_question_idx is not None and selected_question_idx < len(filtered_questions):
                selected_question = filtered_questions[selected_question_idx]
                question_id = selected_question["id"]
                category = selected_question["category"]
                question_data = selected_question["question"]
                
                st.subheader(f"Question: {question_id}")
                
                # Display the question content (not the wrapper JSON)
                with st.expander("Question Details", expanded=True):
                    # Extract the actual question content
                    if "question" in question_data:
                        st.write(question_data["question"])
                    
                    # Show function if available
                    if "function" in question_data:
                        st.write("**Function:**")
                        st.json(question_data["function"])
                
                # Collect results by tool format
                baml_results = []
                fc_results = []
                prompt_results = []
                
                for model_name in data["results"]:
                    if category in data["results"][model_name]:
                        results = data["results"][model_name][category]
                        
                        # Find the result for this specific question
                        question_result = None
                        for result in results:
                            if result.get("id") == question_id:
                                question_result = result
                                break
                        
                        if question_result:
                            # Find the corresponding score
                            score_result = None
                            if model_name in data["models"] and category in data["models"][model_name]:
                                scores = data["models"][model_name][category]
                                for score in scores[1:]:  # Skip overall stats
                                    if score.get("id") == question_id:
                                        score_result = score
                                        break
                            
                            result_data = {
                                "model_name": model_name,
                                "question_result": question_result,
                                "score_result": score_result,
                                "is_valid": score_result.get("valid", True) if score_result else True
                            }
                            
                            if "-baml" in model_name:
                                baml_results.append(result_data)
                            elif "-FC" in model_name:
                                fc_results.append(result_data)
                            else:
                                prompt_results.append(result_data)
                
                # Create tabs for different tool formats
                st.subheader("Model Results")
                
                # Build tab labels with pass/fail indicators
                tab_labels = []
                if baml_results:
                    baml_status = "✅" if any(r["is_valid"] for r in baml_results) else "❌"
                    tab_labels.append(f"BAML {baml_status}")
                if fc_results:
                    fc_status = "✅" if any(r["is_valid"] for r in fc_results) else "❌"
                    tab_labels.append(f"FC {fc_status}")
                if prompt_results:
                    prompt_status = "✅" if any(r["is_valid"] for r in prompt_results) else "❌"
                    tab_labels.append(f"Prompt {prompt_status}")
                
                if tab_labels:
                    tabs = st.tabs(tab_labels)
                    tab_index = 0
                    
                    # BAML tab
                    if baml_results:
                        with tabs[tab_index]:
                            for result_data in baml_results:
                                model_name = result_data["model_name"]
                                question_result = result_data["question_result"]
                                score_result = result_data["score_result"]
                                is_valid = result_data["is_valid"]
                                
                                with st.expander(f"{model_name} {'✅' if is_valid else '❌'}", expanded=True):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Status:**")
                                        status_emoji = "✅" if is_valid else "❌"
                                        st.write(f"{status_emoji} {'Success' if is_valid else 'Failed'}")
                                        
                                        if score_result and "error_type" in score_result:
                                            st.write(f"**Error Type:** {score_result['error_type']}")
                                    
                                    with col2:
                                        st.write("**Execution Result:**")
                                        if "execution_result" in question_result:
                                            st.json(question_result["execution_result"])
                                    
                                    st.write("**Model Output:**")
                                    if "result" in question_result:
                                        st.json(question_result["result"])
                                    
                                    if "model_output" in question_result:
                                        st.write("**Raw Model Output:**")
                                        st.code(str(question_result["model_output"]))
                                    
                                    if score_result and "error" in score_result:
                                        st.write("**Error:**")
                                        error_list = score_result["error"]
                                        if isinstance(error_list, list):
                                            for error_msg in error_list:
                                                st.error(error_msg)
                                        else:
                                            st.error(str(error_list))
                                    
                                    if score_result and "error_details" in score_result:
                                        st.write("**Error Details:**")
                                        st.error(score_result["error_details"])
                        tab_index += 1
                    
                    # FC tab
                    if fc_results:
                        with tabs[tab_index]:
                            for result_data in fc_results:
                                model_name = result_data["model_name"]
                                question_result = result_data["question_result"]
                                score_result = result_data["score_result"]
                                is_valid = result_data["is_valid"]
                                
                                with st.expander(f"{model_name} {'✅' if is_valid else '❌'}", expanded=True):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Status:**")
                                        status_emoji = "✅" if is_valid else "❌"
                                        st.write(f"{status_emoji} {'Success' if is_valid else 'Failed'}")
                                        
                                        if score_result and "error_type" in score_result:
                                            st.write(f"**Error Type:** {score_result['error_type']}")
                                    
                                    with col2:
                                        st.write("**Execution Result:**")
                                        if "execution_result" in question_result:
                                            st.json(question_result["execution_result"])
                                    
                                    st.write("**Model Output:**")
                                    if "result" in question_result:
                                        st.json(question_result["result"])
                                    
                                    if "model_output" in question_result:
                                        st.write("**Raw Model Output:**")
                                        st.code(str(question_result["model_output"]))
                                    
                                    if score_result and "error" in score_result:
                                        st.write("**Error:**")
                                        error_list = score_result["error"]
                                        if isinstance(error_list, list):
                                            for error_msg in error_list:
                                                st.error(error_msg)
                                        else:
                                            st.error(str(error_list))
                                    
                                    if score_result and "error_details" in score_result:
                                        st.write("**Error Details:**")
                                        st.error(score_result["error_details"])
                        tab_index += 1
                    
                    # Prompt tab
                    if prompt_results:
                        with tabs[tab_index]:
                            for result_data in prompt_results:
                                model_name = result_data["model_name"]
                                question_result = result_data["question_result"]
                                score_result = result_data["score_result"]
                                is_valid = result_data["is_valid"]
                                
                                with st.expander(f"{model_name} {'✅' if is_valid else '❌'}", expanded=True):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Status:**")
                                        status_emoji = "✅" if is_valid else "❌"
                                        st.write(f"{status_emoji} {'Success' if is_valid else 'Failed'}")
                                        
                                        if score_result and "error_type" in score_result:
                                            st.write(f"**Error Type:** {score_result['error_type']}")
                                    
                                    with col2:
                                        st.write("**Execution Result:**")
                                        if "execution_result" in question_result:
                                            st.json(question_result["execution_result"])
                                    
                                    st.write("**Model Output:**")
                                    if "result" in question_result:
                                        st.json(question_result["result"])
                                    
                                    if "model_output" in question_result:
                                        st.write("**Raw Model Output:**")
                                        st.code(str(question_result["model_output"]))
                                    
                                    if score_result and "error" in score_result:
                                        st.write("**Error:**")
                                        error_list = score_result["error"]
                                        if isinstance(error_list, list):
                                            for error_msg in error_list:
                                                st.error(error_msg)
                                        else:
                                            st.error(str(error_list))
                                    
                                    if score_result and "error_details" in score_result:
                                        st.write("**Error Details:**")
                                        st.error(score_result["error_details"])
                else:
                    st.info("No results found for this question.")
    else:
        st.info("No test data found. Make sure test data files are in the bfcl_eval/data directory.")


if __name__ == "__main__":
    main()
