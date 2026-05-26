import sys
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

def convert_all_xmls_to_csv(recordings_dir="simulations/recordings"):
    folder = Path(recordings_dir)
    if not folder.exists():
        print(f"Directory {recordings_dir} not found.")
        return

    xml_files = list(folder.glob("*.xml"))
    if not xml_files:
        print(f"No .xml files found in {recordings_dir}")
        return

    for xml_path in xml_files:
        csv_path = xml_path.with_suffix(".csv")
        print(f"Processing: {xml_path.name} -> {csv_path.name}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            records = []
            
            for child in root:
                if child.tag == "tripinfo":
                    records.append({f"tripinfo_{k}": v for k, v in child.attrib.items()})
                    
            if not records:
                print(f"  No tripinfo data found in {xml_path.name}")
                continue
                
            df = pd.DataFrame(records)
            df.to_csv(csv_path, sep=";", index=False)
            print(f"  Success! Saved {csv_path.name}")
            
        except Exception as e:
            print(f"  Error processing {xml_path.name}: {e}")

if __name__ == "__main__":
    convert_all_xmls_to_csv()
