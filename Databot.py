import pandas as pd
import numpy as np
import sys
import argparse
import re
from typing import Dict, List, Any, Tuple, Optional

class CSVChatbot:
    def __init__(self, dataframe: pd.DataFrame, file_path: str):
        self.df = dataframe
        self.file_path = file_path
        self.selected_columns = []
        self.stage = "welcome"
        self.analysis_type = None
        
    def get_column_info(self, column_name: str) -> str:
        """Generate detailed information about a specific column."""
        col = self.df[column_name]
        col_type = col.dtype
        missing = col.isna().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        info = [f"Column: {column_name}"]
        info.append(f"Type: {col_type}")
        info.append(f"Missing values: {missing} ({missing_pct:.2f}%)")
        
        if pd.api.types.is_numeric_dtype(col_type):
            info.append(f"Range: {col.min()} to {col.max()}")
            info.append(f"Mean: {col.mean():.4f}")
            info.append(f"Median: {col.median():.4f}")
            info.append(f"Standard deviation: {col.std():.4f}")
            
            # Check for outliers using IQR method
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            outliers = ((col < (q1 - 1.5 * iqr)) | (col > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                info.append(f"Potential outliers: {outliers} values")
                
        elif pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
            unique_count = col.nunique()
            info.append(f"Unique values: {unique_count}")
            
            if unique_count <= 10:  # For categorical with few values, show all
                value_counts = col.value_counts()
                info.append("Value distribution:")
                for value, count in value_counts.items():
                    pct = (count / len(self.df)) * 100
                    info.append(f"  - {value}: {count} ({pct:.2f}%)")
            else:  # For many values, show top 5
                value_counts = col.value_counts().nlargest(5)
                info.append("Top 5 values:")
                for value, count in value_counts.items():
                    pct = (count / len(self.df)) * 100
                    info.append(f"  - {value}: {count} ({pct:.2f}%)")
        
        elif pd.api.types.is_datetime64_dtype(col_type):
            info.append(f"Date range: {col.min()} to {col.max()}")
            info.append(f"Time span: {(col.max() - col.min()).days} days")
        
        return "\n".join(info)
    
    def get_column_summary(self, column_name: str) -> str:
        """Generate a brief summary of a column."""
        col = self.df[column_name]
        col_type = col.dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            return f"{column_name}: numeric column ranging from {col.min()} to {col.max()}"
        elif pd.api.types.is_datetime64_dtype(col_type):
            return f"{column_name}: date column from {col.min()} to {col.max()}"
        else:
            return f"{column_name}: text column with {col.nunique()} unique values"
    
    def perform_analysis(self, analysis_type: str) -> str:
        """Perform requested analysis on selected columns."""
        if len(self.selected_columns) == 0:
            return "No columns selected. Please select columns first."
            
        if analysis_type == "summary":
            result = ["Summary Statistics:"]
            for col in self.selected_columns:
                result.append("\n" + self.get_column_info(col))
            return "\n".join(result)
            
        elif analysis_type == "correlation":
            numeric_cols = [col for col in self.selected_columns 
                          if pd.api.types.is_numeric_dtype(self.df[col].dtype)]
            
            if len(numeric_cols) < 2:
                return "Correlation analysis requires at least 2 numeric columns. Please select more numeric columns."
                
            corr_matrix = self.df[numeric_cols].corr()
            result = ["Correlation Matrix:"]
            result.append("\n" + str(corr_matrix.round(4)))
            
            # Find and report strongest correlations
            corr_unstack = corr_matrix.unstack()
            corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
            
            if not corr_unstack.empty:
                strong_corr = corr_unstack[abs(corr_unstack) > 0.5]
                if not strong_corr.empty:
                    result.append("\nStrong relationships:")
                    for (col1, col2), corr_val in strong_corr.items():
                        direction = "positive" if corr_val > 0 else "negative"
                        strength = "very strong" if abs(corr_val) > 0.8 else "strong"
                        result.append(f"- {col1} and {col2}: {strength} {direction} correlation ({corr_val:.4f})")
            
            return "\n".join(result)
            
        elif analysis_type == "distribution":
            result = ["Distribution Analysis:"]
            
            for col in self.selected_columns:
                if pd.api.types.is_numeric_dtype(self.df[col].dtype):
                    # Get quantiles and distribution shape
                    quantiles = self.df[col].quantile([0.25, 0.5, 0.75])
                    skew = self.df[col].skew()
                    
                    result.append(f"\n{col}:")
                    result.append(f"- Range: {self.df[col].min()} to {self.df[col].max()}")
                    result.append(f"- Quartiles: Q1={quantiles[0.25]:.4f}, Q2={quantiles[0.5]:.4f}, Q3={quantiles[0.75]:.4f}")
                    
                    if abs(skew) < 0.5:
                        shape = "approximately symmetric"
                    elif skew > 0:
                        shape = f"positively skewed ({skew:.4f})"
                    else:
                        shape = f"negatively skewed ({skew:.4f})"
                    result.append(f"- Distribution shape: {shape}")
                    
                elif pd.api.types.is_categorical_dtype(self.df[col].dtype) or pd.api.types.is_object_dtype(self.df[col].dtype):
                    # For categorical columns, show frequency distribution
                    value_counts = self.df[col].value_counts().nlargest(10)
                    total = len(self.df)
                    
                    result.append(f"\n{col}:")
                    result.append(f"- {self.df[col].nunique()} unique values")
                    result.append("- Top values by frequency:")
                    
                    for value, count in value_counts.items():
                        pct = (count / total) * 100
                        result.append(f"  * {value}: {count} ({pct:.2f}%)")
            
            return "\n".join(result)
            
        elif analysis_type == "missing":
            result = ["Missing Value Analysis:"]
            
            for col in self.selected_columns:
                missing = self.df[col].isna().sum()
                missing_pct = (missing / len(self.df)) * 100
                
                if missing > 0:
                    result.append(f"\n{col}:")
                    result.append(f"- Missing values: {missing} ({missing_pct:.2f}%)")
                    
                    # For numeric columns, compare stats of rows with/without missing values
                    if pd.api.types.is_numeric_dtype(self.df[col].dtype):
                        other_cols = [c for c in self.selected_columns if c != col and pd.api.types.is_numeric_dtype(self.df[c].dtype)]
                        
                        if other_cols:
                            result.append("- Related patterns:")
                            for other_col in other_cols[:3]:  # Limit to avoid too much output
                                with_missing = self.df[self.df[col].isna()][other_col].mean()
                                without_missing = self.df[~self.df[col].isna()][other_col].mean()
                                
                                if not pd.isna(with_missing) and not pd.isna(without_missing):
                                    diff_pct = abs(with_missing - without_missing) / without_missing * 100
                                    if diff_pct > 10:  # Only report if there's a notable difference
                                        result.append(f"  * Rows with missing {col} have {with_missing:.2f} average {other_col} " +
                                                     f"vs {without_missing:.2f} for non-missing rows ({diff_pct:.1f}% difference)")
            
            if len(result) == 1:
                result.append("\nNo missing values found in the selected columns.")
                
            return "\n".join(result)
        
        elif analysis_type == "time":
            # Check if we have any datetime columns
            datetime_cols = [col for col in self.selected_columns 
                           if pd.api.types.is_datetime64_dtype(self.df[col].dtype)]
            
            if not datetime_cols:
                return "Time series analysis requires date/time columns. None were found in your selection."
            
            result = ["Time Series Analysis:"]
            
            for date_col in datetime_cols:
                result.append(f"\nAnalysis using {date_col} as time dimension:")
                result.append(f"- Time range: {self.df[date_col].min()} to {self.df[date_col].max()}")
                
                # Find numeric columns to analyze over time
                numeric_cols = [col for col in self.selected_columns 
                              if col != date_col and pd.api.types.is_numeric_dtype(self.df[col].dtype)]
                
                if numeric_cols:
                    for num_col in numeric_cols[:3]:  # Limit to 3 to avoid excessive output
                        # Group by year and month for trend analysis
                        try:
                            self.df['year_month'] = self.df[date_col].dt.to_period('M')
                            monthly_means = self.df.groupby('year_month')[num_col].mean()
                            
                            result.append(f"\n{num_col} trends:")
                            
                            # Get overall trend direction
                            first_months = monthly_means.head(3).mean()
                            last_months = monthly_means.tail(3).mean()
                            
                            if last_months > first_months * 1.05:
                                trend = "increasing"
                            elif last_months < first_months * 0.95:
                                trend = "decreasing"
                            else:
                                trend = "stable"
                                
                            result.append(f"- Overall trend: {trend}")
                            result.append(f"- Starting value (avg of first 3 months): {first_months:.4f}")
                            result.append(f"- Ending value (avg of last 3 months): {last_months:.4f}")
                            
                            # Calculate month-over-month changes
                            monthly_pct_change = monthly_means.pct_change() * 100
                            avg_monthly_change = monthly_pct_change.mean()
                            
                            result.append(f"- Average month-over-month change: {avg_monthly_change:.2f}%")
                            
                        except Exception as e:
                            result.append(f"- Error analyzing {num_col} over time: {str(e)}")
                        finally:
                            # Clean up temporary column
                            if 'year_month' in self.df.columns:
                                self.df.drop('year_month', axis=1, inplace=True)
                
            return "\n".join(result)
            
        elif analysis_type == "custom":
            # This is a placeholder for custom analysis that can be expanded later
            return "Custom analysis is not implemented yet. Please choose another analysis type."
        
        else:
            return f"Analysis type '{analysis_type}' not recognized."
    
    def process_input(self, user_input: str) -> str:
        """Process user input based on the current stage of the conversation."""
        user_input = user_input.lower().strip()
        
        # Handle universal commands
        if user_input in ['exit', 'quit', 'bye', 'goodbye']:
            self.stage = "exit"
            return "exit"
            
        if user_input in ['help', '?']:
            return self.get_help()
            
        if user_input in ['restart', 'reset', 'start over']:
            self.stage = "welcome"
            self.selected_columns = []
            self.analysis_type = None
            return "Conversation restarted. Let's begin again."
        
        # Process based on current conversation stage
        if self.stage == "welcome":
            self.stage = "column_selection"
            columns_list = "\n".join([f"{i+1}. {col}" for i, col in enumerate(self.df.columns)])
            return (f"I've analyzed your CSV file: {self.file_path}\n\n"
                   f"This dataset contains {len(self.df)} rows and {len(self.df.columns)} columns.\n\n"
                   f"Available columns:\n{columns_list}\n\n"
                   f"Which columns would you like to explore? You can specify them by number, name, "
                   f"or type 'all' for all columns. Multiple selections can be separated by commas.")
                   
        elif self.stage == "column_selection":
            selected = self._parse_column_selection(user_input)
            
            if not selected:
                return ("I couldn't identify the columns you specified. Please use column numbers, names, "
                       "or 'all', separated by commas.")
                
            self.selected_columns = selected
            self.stage = "column_info"
            
            columns_summary = "\n".join([f"- {self.get_column_summary(col)}" for col in self.selected_columns])
            
            return (f"You've selected {len(self.selected_columns)} columns:\n{columns_summary}\n\n"
                   f"Would you like to see detailed information about these columns? (yes/no)")
                   
        elif self.stage == "column_info":
            if user_input.startswith('y'):
                columns_info = "\n\n".join([self.get_column_info(col) for col in self.selected_columns])
                self.stage = "analysis_selection"
                
                return (f"Here's detailed information about your selected columns:\n\n{columns_info}\n\n"
                       f"What type of analysis would you like to perform?\n"
                       f"1. Summary statistics\n"
                       f"2. Correlation analysis\n"
                       f"3. Distribution analysis\n"
                       f"4. Missing value analysis\n"
                       f"5. Time series analysis (if date columns are available)\n"
                       f"6. Select different columns\n"
                       f"You can specify by number or name.")
            else:
                self.stage = "analysis_selection"
                
                return (f"What type of analysis would you like to perform on your selected columns?\n"
                       f"1. Summary statistics\n"
                       f"2. Correlation analysis\n"
                       f"3. Distribution analysis\n"
                       f"4. Missing value analysis\n"
                       f"5. Time series analysis (if date columns are available)\n"
                       f"6. Select different columns\n"
                       f"You can specify by number or name.")
                       
        elif self.stage == "analysis_selection":
            analysis_type = self._parse_analysis_selection(user_input)
            
            if analysis_type == "invalid":
                return ("I didn't understand your selection. Please specify an analysis type by number or name, "
                       "or type 'help' to see options.")
                       
            if analysis_type == "column_reselection":
                self.stage = "column_selection"
                columns_list = "\n".join([f"{i+1}. {col}" for i, col in enumerate(self.df.columns)])
                return (f"Available columns:\n{columns_list}\n\n"
                       f"Which columns would you like to explore? You can specify them by number, name, "
                       f"or type 'all' for all columns. Multiple selections can be separated by commas.")
            
            self.analysis_type = analysis_type
            result = self.perform_analysis(analysis_type)
            self.stage = "follow_up"
            
            return (f"{result}\n\n"
                   f"What would you like to do next?\n"
                   f"1. Perform another analysis on the same columns\n"
                   f"2. Select different columns\n"
                   f"3. Exit")
                   
        elif self.stage == "follow_up":
            if '1' in user_input or 'another' in user_input or 'analysis' in user_input:
                self.stage = "analysis_selection"
                
                return (f"What type of analysis would you like to perform on your selected columns?\n"
                       f"1. Summary statistics\n"
                       f"2. Correlation analysis\n"
                       f"3. Distribution analysis\n"
                       f"4. Missing value analysis\n"
                       f"5. Time series analysis (if date columns are available)\n"
                       f"6. Select different columns\n"
                       f"You can specify by number or name.")
                       
            elif '2' in user_input or 'select' in user_input or 'different' in user_input or 'columns' in user_input:
                self.stage = "column_selection"
                columns_list = "\n".join([f"{i+1}. {col}" for i, col in enumerate(self.df.columns)])
                return (f"Available columns:\n{columns_list}\n\n"
                       f"Which columns would you like to explore? You can specify them by number, name, "
                       f"or type 'all' for all columns. Multiple selections can be separated by commas.")
                       
            elif '3' in user_input or 'exit' in user_input or 'quit' in user_input:
                self.stage = "exit"
                return "exit"
                
            else:
                return ("I didn't understand your selection. Please choose one of the options or type 'help' "
                       "for assistance.")
        
        return "I'm not sure how to proceed. Type 'help' for assistance or 'restart' to start over."
    
    def _parse_column_selection(self, selection_text: str) -> List[str]:
        """Parse user input to identify selected columns."""
        if selection_text.lower() == 'all':
            return list(self.df.columns)
            
        selected_columns = []
        
        # Split by commas and handle each part
        parts = [part.strip() for part in selection_text.split(',')]
        
        for part in parts:
            # Check if it's a number
            if part.isdigit():
                idx = int(part) - 1  # Convert to 0-based index
                if 0 <= idx < len(self.df.columns):
                    selected_columns.append(self.df.columns[idx])
            # Check if it's a column name or part of it
            else:
                matches = [col for col in self.df.columns if part.lower() in col.lower()]
                selected_columns.extend(matches)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(selected_columns))
    
    def _parse_analysis_selection(self, selection_text: str) -> str:
        """Parse user input to identify selected analysis type."""
        selection_text = selection_text.lower()
        
        if '1' in selection_text or 'summary' in selection_text or 'stat' in selection_text:
            return "summary"
        elif '2' in selection_text or 'corr' in selection_text or 'relation' in selection_text:
            return "correlation"
        elif '3' in selection_text or 'distri' in selection_text:
            return "distribution"
        elif '4' in selection_text or 'missing' in selection_text or 'null' in selection_text:
            return "missing"
        elif '5' in selection_text or 'time' in selection_text or 'series' in selection_text:
            return "time"
        elif '6' in selection_text or 'select' in selection_text or 'column' in selection_text or 'different' in selection_text:
            return "column_reselection"
        else:
            return "invalid"
    
    def get_help(self) -> str:
        """Generate a help message based on the current stage."""
        common_help = [
            "Common commands:",
            "- 'restart': Start over from the beginning",
            "- 'help': Show this help message",
            "- 'exit', 'quit', 'bye': Exit the chatbot"
        ]
        
        if self.stage == "welcome" or self.stage == "column_selection":
            specific_help = [
                "At this stage, you can select columns to analyze by:",
                "- Specifying column numbers (e.g., '1, 3, 5')",
                "- Specifying column names (e.g., 'age, income')",
                "- Typing 'all' to select all columns",
                "Multiple selections can be separated by commas."
            ]
        elif self.stage == "column_info":
            specific_help = [
                "At this stage, you can:",
                "- Type 'yes' to see detailed information about selected columns",
                "- Type 'no' to skip this and go directly to choosing an analysis type"
            ]
        elif self.stage == "analysis_selection":
            specific_help = [
                "At this stage, you can select an analysis type by:",
                "- Typing the number (1-6)",
                "- Typing the name (e.g., 'correlation', 'summary')",
                "",
                "Available analyses:",
                "1. Summary statistics: Basic statistics for each column",
                "2. Correlation: Relationships between numeric columns",
                "3. Distribution: How values are distributed in each column",
                "4. Missing values: Analysis of null or missing data",
                "5. Time series: Trends over time (requires date columns)",
                "6. Select different columns: Go back to column selection"
            ]
        elif self.stage == "follow_up":
            specific_help = [
                "At this stage, you can:",
                "1. Perform another analysis on the same columns",
                "2. Select different columns to analyze",
                "3. Exit the chatbot"
            ]
        else:
            specific_help = ["Type 'restart' to start over."]
        
        return "\n".join(["Help:"] + specific_help + [""] + common_help)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive multi-stage CSV data chatbot.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    return parser.parse_args()

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file and convert date columns if possible."""
    try:
        # First pass: load as strings to identify potential date columns
        df = pd.read_csv(file_path, low_memory=False)
        
        # Try to convert date-like columns to datetime
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if column might contain dates
                try:
                    # Try to parse a sample of values
                    sample = df[column].dropna().head(100)
                    if len(sample) > 0:
                        pd.to_datetime(sample, errors='raise')
                        # If successful, convert the whole column
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                except:
                    # Not a date column, leave as is
                    pass
        
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def main():
    """Main function to run the CSV chatbot."""
    args = parse_arguments()
    
    print(f"Loading CSV file: {args.file_path}")
    df = load_csv(args.file_path)
    
    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    
    print("\n" + "="*60)
    print("Welcome to the Multi-Stage CSV Data Chatbot!")
    print("I'll guide you through exploring and analyzing your CSV data.")
    print("Type 'help' at any time for assistance or 'exit' to quit.")
    print("="*60)
    
    chatbot = CSVChatbot(df, args.file_path)
    
    # Start the conversation
    prompt = chatbot.process_input("start")
    print("\n" + prompt)
    
    while True:
        user_input = input("\n> ")
        response = chatbot.process_input(user_input)
        
        if response == "exit":
            print("Goodbye! Thanks for using the CSV Data Chatbot.")
            break
            
        print("\n" + response)

if __name__ == "__main__":
    main()