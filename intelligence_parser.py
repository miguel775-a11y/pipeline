import json
import pandas as pd
import sys

def extract_data(data):
    entries = []
    if isinstance(data, dict):
        if 'nse_code' in data:
            entry = {'nse_code': data['nse_code']}
            if 'target_price' in data:
                entry['target_price'] = data['target_price']
            if 'universal_quantitative_estimates' in data:
                uqe = data['universal_quantitative_estimates']
                if isinstance(uqe, dict):
                    for k1, v1 in uqe.items():
                        if isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                entry[f"{k1}_{k2}"] = v2
                        else:
                            entry[f"{k1}"] = v1
            entries.append(entry)
        for val in data.values():
            entries.extend(extract_data(val))
    elif isinstance(data, list):
        for val in data:
            entries.extend(extract_data(val))
    return entries

def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else ['ABBIND_broker_intelligence.json', 'macro_state.json']
    all_entries = []

    for f in files:
        try:
            with open(f, 'r') as file:
                parsed_json = json.load(file)
                all_entries.extend(extract_data(parsed_json))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if all_entries:
        df = pd.DataFrame(all_entries)
        if 'nse_code' in df.columns:
            df.set_index('nse_code', inplace=True)
        print(df)
    else:
        print("No valid data found.")

if __name__ == '__main__':
    main()
