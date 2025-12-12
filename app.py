import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from io import StringIO, BytesIO
import colorsys
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from itertools import combinations

# --- PAGE CONFIG ---
st.set_page_config(page_title="Material Analysis Dashboard", layout="wide")
st.title("🧪 Material Analysis Dashboard")
st.markdown("Upload your experiment CSV files (both the main file and RH variants) to generate the report.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Settings")
    plot_top_num = st.number_input("Top Samples to Plot (0 = All)", min_value=0, value=5)
    exclude_pu_flag = st.checkbox("Exclude PU Samples", value=False)
    # File Uploader
    uploaded_files_list = st.file_uploader("Upload CSV Files", type='csv', accept_multiple_files=True)

# --- HELPER FUNCTIONS ---
# (These are your original functions, slightly modified to read from memory instead of disk)

def split_csv_by_comma_lines(content_string):
    lines = content_string.splitlines()[1:]  # Skip metadata line
    blocks, current_block = [], []
    for line in lines:
        if not any(char.isalnum() for char in line):
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line + '\n') # Add newline back for consistency
    if current_block:
        blocks.append(current_block)
    return blocks

def parse_table_blocks(blocks, default_name="Name"):
    tables = []
    for i, block in enumerate(blocks):
        try:
            df = pd.read_csv(StringIO("".join(block)), header=None)
            raw_header = df.iloc[0].tolist()
            fixed_header = [default_name if not str(col).strip() else str(col).strip() for col in raw_header]
            df = df[1:].reset_index(drop=True)
            df.columns = fixed_header
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            tables.append(df)
        except Exception as e:
            pass 
    return tables

def extract_top_and_bottom_tables(file_content):
    # Convert bytes to string
    content_string = file_content.decode('utf-8')
    blocks = split_csv_by_comma_lines(content_string)
    all_tables = parse_table_blocks(blocks)
    
    if not all_tables: 
        return pd.DataFrame(), pd.DataFrame()

    top_table = all_tables[0]
    bottom_df = all_tables[-1]

    # Use row 0 as new header for bottom table
    bottom_df = bottom_df.transpose().reset_index(drop=False)
    new_header = bottom_df.iloc[0]
    bottom_df = bottom_df[1:]
    bottom_df.columns = new_header
    
    # Safely rename first column
    cols = bottom_df.columns.values
    cols[0] = "Strain (%)"
    bottom_df.columns = cols

    numeric_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    def safe_float_convert(val):
        if isinstance(val, str) and numeric_pattern.match(val.strip()):
            return float(val)
        return val

    top_table = top_table.applymap(safe_float_convert)
    return top_table, bottom_df

def clean_percent(col):
    return pd.to_numeric(col.astype(str).str.rstrip('%'), errors='coerce')

def normalise_description(df):
    if 'Description' not in df.columns:
        df['Description'] = np.nan
    if 'Label' in df.columns:
        df['Description'] = df['Description'].fillna(df['Label'])
        df = df.drop(columns=['Label'])
    return df

# --- MAIN LOGIC ---

if uploaded_files_list:
    # 1. ORGANIZE FILES
    # We create a dictionary to mimic your file system: {'V870.csv': file_object, ...}
    file_map = {f.name: f.getvalue() for f in uploaded_files_list}
    
    # Identify "Main" experiment files (those that don't end in RH.csv)
    # Adjust regex to find files like V870.csv but NOT V870-75RH.csv
    exp_codes = []
    for fname in file_map.keys():
        # Heuristic: if it has 'RH' in the name, it's a variant, not a parent
        if 'RH' not in fname and fname.lower().endswith('.csv'):
            exp_codes.append(fname.replace('.csv', ''))
    
    if not exp_codes:
        st.warning("No main experiment files found (files without 'RH' in the name).")
        st.stop()

    # Initialize Containers
    df_results = pd.DataFrame()
    df_curves_all = {}
    rhnum_list = ['50', '75'] # Default list

    # 2. PROCESS DATA
    progress_bar = st.progress(0)
    
    for idx, exp_name in enumerate(exp_codes):
        main_filename = f"{exp_name}.csv"
        
        if main_filename not in file_map:
            continue
            
        # Process Main File
        df_res_i, df_cur_i = extract_top_and_bottom_tables(file_map[main_filename])
        
        if df_res_i.empty:
            continue

        df_res_i['OrigName'] = df_res_i['Name']
        df_res_i = normalise_description(df_res_i)
        
        # Extract RH from name or default to 50
        df_res_i['RH'] = df_res_i['Name'].str.extract(r'(?i)(..)(?=RH)').fillna('50') + 'RH'
        
        # Add new unique values to rhnum_list
        unique_rhs = df_res_i['RH'].unique()
        for val in unique_rhs:
            clean_rh = val.replace('RH','')
            if clean_rh not in rhnum_list:
                rhnum_list.append(clean_rh)

        # Check for secondary files in the upload list
        for rhnum_str in rhnum_list:
            for rh_str in [f'-{rhnum_str}RH', f'.{rhnum_str}RH', f'.{rhnum_str}rh', f'-{rhnum_str}rh']:
                sec_fname = f"{exp_name}{rh_str}.csv"
                if sec_fname in file_map:
                    df_res_rh, df_cur_rh = extract_top_and_bottom_tables(file_map[sec_fname])
                    df_res_rh['OrigName'] = df_res_rh['Name']
                    df_res_rh = normalise_description(df_res_rh)
                    df_res_rh['RH'] = f'{rhnum_str}RH'
                    
                    df_res_i = pd.concat([df_res_i, df_res_rh], ignore_index=True)
                    df_curves_all[exp_name + '-' + rhnum_str + 'RH'] = df_cur_rh

        df_res_i['ExpCode'] = exp_name
        df_results = pd.concat([df_results, df_res_i], ignore_index=True)
        df_curves_all[exp_name] = df_cur_i
        
        progress_bar.progress((idx + 1) / len(exp_codes))

    # 3. CLEANING & CALCULATIONS
    if df_results.empty:
        st.error("Could not parse data from uploaded files.")
        st.stop()

    rename_map = {
        'UTS AVG': 'UTS (MPa)', 'UTS STD': 'errUTS (MPa)',
        'Elongation AVG': 'Elongation (%)', 'Elongation STD': 'errElongation (%)',
        'Youngs AVG': 'Modulus (MPa)', 'Youngs STD (lol)': 'errModulus (MPa)',
        'Toughness AVG': 'Toughness (MPa)', 'Toughness STD': 'errToughness (MPa)'
    }
    df_results = df_results.rename(columns=rename_map)

    percent_cols = [col for col in df_results.columns if '%' in col]
    df_results[percent_cols] = df_results[percent_cols].apply(clean_percent)

    # Standardize Names
    rhnum_list = (df_results['RH'].str.replace('RH', '', regex=False).unique().tolist())
    for rhnum_str in rhnum_list:
        for rh_str in [f'.{rhnum_str}RH', f'.{rhnum_str}rh', f'-{rhnum_str}rh']:
            df_results['Name'] = df_results['Name'].str.replace(rh_str, f'-{rhnum_str}RH')
    df_results['Name'] = df_results['Name'].str.replace('-50RH', '')

    # Fill Descriptions
    mask = df_results['Description'].isna()
    df_results.loc[mask, 'Description'] = df_results.loc[mask, 'Name']
    
    # Create Sample Code
    df_results['Sample'] = np.where(
        df_results['Name'].str.match(r'^V\d{3}'),
        df_results['Name'].str[4:],
        df_results['Name']
    )
    for rhnum in rhnum_list:
        clean_str = f'-{rhnum}RH'
        df_results['Description'] = df_results['Description'].str.replace(clean_str, '')
        df_results['Sample'] = df_results['Sample'].str.replace(clean_str, '')
    
    df_results['SampleCode'] = df_results['ExpCode'] + df_results['Sample']
    
    # Filter PU
    if exclude_pu_flag:
         df_results = df_results[~df_results['Description'].fillna('').astype(str).str.startswith('PU')]

    # 4. MASTER CURVES INTERPOLATION
    x_master = np.arange(0, 800, 0.0819075)
    df_curves_master = pd.DataFrame({'Strain (%)': x_master})
    
    # Build dictionary for plotting
    data_dict = {}
    origname_to_name = dict(zip(df_results['OrigName'].dropna().astype(str), df_results['Name'].dropna().astype(str)))

    for key, df_i in df_curves_all.items():
        xp = pd.to_numeric(df_i['Strain (%)'], errors='coerce')
        for col in df_i.columns:
            if col == 'Strain (%)': continue
            fp = pd.to_numeric(df_i[col], errors='coerce')
            
            # Clean for Master
            mask = ~(xp.isna() | fp.isna())
            if mask.sum() > 0:
                # Add to data_dict for plotting
                clean_df = pd.DataFrame({'X-axis meas': xp[mask], 'Y-axis meas': fp[mask]})
                # Try to map original column name to new cleaned name
                new_name = col
                for oname, clean_name in origname_to_name.items():
                    if oname in col:
                        new_name = clean_name
                        break
                data_dict[new_name] = clean_df

    # 5. VISUALIZATION
    
    # --- A. BAR CHARTS ---
    st.subheader("📊 Mechanical Properties Comparison")
    
    # Filter for plotting
    df_filtered = df_results.drop_duplicates(subset='Name')
    samples = sorted(df_filtered['SampleCode'].unique())
    groups = sorted(df_filtered['RH'].unique())
    
    plot_metrics = ['Modulus (MPa)', 'Elongation (%)', 'UTS (MPa)', 'Toughness (MPa)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    bar_width = 0.35
    x = np.arange(len(samples))
    
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        err_metric = f'err{metric}' if f'err{metric}' in df_filtered.columns else None
        
        for i, group in enumerate(groups):
            group_df = df_filtered[df_filtered['RH'] == group]
            group_df = group_df.set_index('SampleCode').reindex(samples).reset_index()
            
            # Use 0 for missing values to align bars
            vals = group_df[metric].fillna(0)
            errs = group_df[err_metric].fillna(0) if err_metric else 0
            
            pos = x + (i - len(groups)/2) * bar_width + bar_width/2
            ax.bar(pos, vals, bar_width, yerr=errs, label=group, capsize=5, alpha=0.8)
            
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(x)
        if idx >= 2:
            ax.set_xticklabels(samples, rotation=90)
        else:
            ax.set_xticklabels([])
            
        if idx == 1: ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig)

    # --- B. PERCENT CHANGE ---
    st.subheader("📈 Percentage Change")
    
    metrics_calc = ['UTS (MPa)', 'Elongation (%)', 'Modulus (MPa)', 'Toughness (MPa)']
    pivoted = df_filtered.pivot(index='SampleCode', columns='RH', values=metrics_calc)
    
if len(groups) >= 2:
        # Calculate % Change between first two groups (e.g., 50RH to 75RH)
        g1, g2 = groups[0], groups[1]
        pct_df = pd.DataFrame(index=pivoted.index)
        
        for m in metrics_calc:
            try:
                col_a, col_b = (m, g1), (m, g2)
                # Pivot creates MultiIndex, access strictly
                if col_a in pivoted.columns and col_b in pivoted.columns:
                     pct_df[f"{m} Δ%"] = 100 * (pivoted[col_b] / pivoted[col_a] - 1)
            except: pass
            
        st.dataframe(pct_df.style.format("{:.2f}%"))
    else:
        st.info("Need at least 2 RH groups to calculate percentage change.")

    # --- C. RAW CURVES ---
    st.subheader("〰️ Stress-Strain Curves")
    
    # Sort by Toughness Retention if possible
    plot_codes = samples
    if len(groups) >= 2 and plot_top_num > 0:
        # Try to find toughness retention
        try:
            toughness_col = 'Toughness (MPa)'
            g1, g2 = groups[0], groups[1]
            # Re-calculate specific ratio for sorting
            piv_tough = df_filtered.pivot(index='SampleCode', columns='RH', values=toughness_col)
            ratio_series = piv_tough[g2] / piv_tough[g1]
            plot_codes = ratio_series.sort_values(ascending=False).head(plot_top_num).index.tolist()
        except:
            plot_codes = samples[:plot_top_num] # Fallback
    elif plot_top_num > 0:
        plot_codes = samples[:plot_top_num]
        
    fig_curve, ax_curve = plt.subplots(figsize=(14, 8))
    
    # Generate Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_codes)))
    color_map = dict(zip(plot_codes, colors))
    
    # Plot Solid Lines (First Group / 50RH)
    plotted_labels = []
    
    # Sort names to prioritize plot_codes
    all_curve_names = sorted(data_dict.keys())
    
    for name in all_curve_names:
        # Find which sample code this belongs to
        base_code = None
        for code in plot_codes:
            if code in name:
                base_code = code
                break
        
        if not base_code: continue # Skip if not in top list
        
        df_c = data_dict[name]
        is_secondary = '-75RH' in name or '-90RH' in name # Heuristic for dashed
        
        linestyle = '--' if is_secondary else '-'
        alpha = 0.6 if is_secondary else 1.0
        label = base_code if not is_secondary else "_nolegend_"
        
        # Avoid duplicate labels
        if label in plotted_labels: label = "_nolegend_"
        else: plotted_labels.append(label)
            
        ax_curve.plot(df_c['X-axis meas'], df_c['Y-axis meas'], 
                      color=color_map[base_code], linestyle=linestyle, 
                      linewidth=2, label=label, alpha=alpha)

    ax_curve.set_xlabel("Strain (%)")
    ax_curve.set_ylabel("Stress (MPa)")
    ax_curve.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_curve.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig_curve)

    # --- D. EXCEL DOWNLOAD ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Results')
        if 'pct_df' in locals():
            pct_df.to_excel(writer, sheet_name='Pct Change')
            
    st.download_button(
        label="📥 Download Analysis Excel",
        data=output.getvalue(),
        file_name="Material_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👆 Please upload CSV files in the sidebar to begin.")
