import json
import pandas as pd

def flatten_dict(d, parent_key='', sep='_'):
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_from_json(file_path):
    """Extracts target_price and universal_quantitative_estimates from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    def find_records(node):
        records = []
        if isinstance(node, dict):
            has_target = 'target_price' in node
            has_uqe = 'universal_quantitative_estimates' in node

            if has_target or has_uqe:
                record = {}
                if 'nse_code' in node:
                    record['nse_code'] = node['nse_code']
                elif 'company_name' in node:
                    record['company_name'] = node['company_name']

                if has_target:
                    record['target_price'] = node['target_price']

                if has_uqe:
                    uqe_flat = flatten_dict(node['universal_quantitative_estimates'])
                    record.update(uqe_flat)

                records.append(record)

            for v in node.values():
                records.extend(find_records(v))
        elif isinstance(node, list):
            for item in node:
                records.extend(find_records(item))
        return records

    return find_records(data)

def main():
    files = ['ABBIND_broker_intelligence.json', 'macro_state.json']
    all_records = []
    for file in files:
        all_records.extend(extract_from_json(file))

    if all_records:
        df = pd.DataFrame(all_records)
        print(df.to_string(index=False))
    else:
        print("No records found.")

if __name__ == '__main__':
    main()
